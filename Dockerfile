# Image reproductible du service tournees-api, avec le binaire VROOM local.
#
# Le service reste UN SEUL conteneur : Flask/Gunicorn devant, binaire VROOM
# appele en subprocess derriere. VROOM n'ecoute sur aucun port et n'est jamais
# expose ; seul Gunicorn ecoute, sur $PORT, exactement comme aujourd'hui.
#
# VROOM est compile PENDANT LE BUILD, jamais au demarrage : un service Render
# qui compile a chaque boot serait lent, fragile et non reproductible.

# La meme base sert au build et au runtime : glibc et libstdc++ identiques,
# donc le binaire compile a l'etape 1 est garanti compatible a l'etape 2.
# Digest epingle : le tag seul peut etre repointe en amont.
ARG PYTHON_IMAGE=python:3.14.3-slim-trixie@sha256:5e59aae31ff0e87511226be8e2b94d78c58f05216efda3b07dbbed938ec8583b


# =============================================================================
# ETAPE 1 — COMPILATION DE VROOM
# =============================================================================
FROM ${PYTHON_IMAGE} AS vroom-build

# Version epinglee par tag ET par SHA de commit : un tag peut etre deplace,
# un SHA non. Le build echoue si les deux ne concordent plus.
ARG VROOM_VERSION=v1.15.0
ARG VROOM_COMMIT=43dd7d0b8b560431eb555bf335cf4797eb7343c4

# Compilation SEQUENTIELLE par defaut. `-j$(nproc)` lancait autant de g++ que
# le builder declare de coeurs ; chaque g++ compilant du C++20 avec -O3 pese
# plusieurs centaines de mega-octets, et le build Render mourait par manque de
# memoire avant meme d'installer les dependances Python. Le nombre de coeurs
# visibles n'a rien a voir avec la memoire disponible : c'est cette memoire
# qui borne le parallelisme, donc elle seule doit le fixer.
# Ne relever cette valeur que sur un builder dont on connait la RAM.
ARG VROOM_BUILD_JOBS=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libasio-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build/vroom

# Les trois sous-modules sont initialises NOMMEMENT, sans --recursive :
# --recurse-submodules descendait aussi dans les sous-modules de rapidjson et
# ramenait thirdparty/gtest, dont aucun test n'est execute ici. Les trois
# chemins ci-dessous sont exactement ceux declares par le .gitmodules de
# VROOM ; en ajouter ou en retirer ferait echouer la compilation, pas passer
# silencieusement.
RUN git clone --branch "${VROOM_VERSION}" --depth 1 \
        https://github.com/VROOM-Project/vroom.git . \
 && test "$(git rev-parse HEAD)" = "${VROOM_COMMIT}" \
 && git submodule update --init --depth 1 \
        include/cxxopts include/polylineencoder include/rapidjson \
 && echo "vroom ${VROOM_VERSION} @ ${VROOM_COMMIT}"

# USE_ROUTING=false retire du binaire les wrappers OSRM / ORS / Valhalla.
# Ce n'est pas une optimisation de taille : c'est la garantie STRUCTURELLE que
# ce binaire ne peut pas emettre de requete reseau. Les seuls appels externes
# du service restent les appels Matrix ORS faits par Python. Nous fournissons
# toujours une matrice personnalisee, donc aucun routeur n'est necessaire.
# libglpk est volontairement absent : il ne sert qu'au mode -c (choose-eta),
# que nous n'utilisons pas, et il ajouterait une dependance au runtime.
RUN make -C src -j"${VROOM_BUILD_JOBS}" USE_ROUTING=false

# Verification a l'etape de build : un binaire muet ici ne doit jamais
# atteindre l'image finale.
RUN ./bin/vroom --version \
 && ./bin/vroom --version > /build/vroom-version.txt \
 && ldd ./bin/vroom > /build/vroom-ldd.txt \
 && cat /build/vroom-ldd.txt


# =============================================================================
# ETAPE 2 — IMAGE FINALE
# =============================================================================
FROM ${PYTHON_IMAGE} AS runtime

ARG VROOM_VERSION=v1.15.0
ARG VROOM_COMMIT=43dd7d0b8b560431eb555bf335cf4797eb7343c4

# On ne copie QUE le binaire, sa licence et sa version. Aucun compilateur,
# aucune source, aucun submodule ne survit dans l'image finale.
COPY --from=vroom-build /build/vroom/bin/vroom      /usr/local/bin/vroom
COPY --from=vroom-build /build/vroom/LICENSE        /usr/local/share/vroom/LICENSE
COPY --from=vroom-build /build/vroom-version.txt    /usr/local/share/vroom/VERSION
COPY --from=vroom-build /build/vroom-ldd.txt        /usr/local/share/vroom/LDD

# Aucune bibliotheque supplementaire n'est copiee : USE_ROUTING=false supprime
# la dependance a libssl, et libstdc++ / libgcc viennent de la base commune.
# Ce RUN echoue le build si ce n'etait pas vrai.
RUN /usr/local/bin/vroom --version

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOCAL_VROOM_BINARY=/usr/local/bin/vroom \
    LOCAL_VROOM_VERSION=${VROOM_VERSION} \
    LOCAL_VROOM_COMMIT=${VROOM_COMMIT}

WORKDIR /app

# Les dependances Python d'abord : cette couche ne change pas quand app.py
# change, donc les rebuilds restent rapides.
COPY requirements-docker.txt ./
RUN pip install -r requirements-docker.txt

COPY app.py local_vroom.py ./
COPY tools/ ./tools/

# Repertoire de travail des fichiers temporaires VROOM. Il est explicite et
# possede par l'utilisateur applicatif : le wrapper y ecrit et y nettoie, sans
# jamais dependre d'un /tmp partage.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /var/tmp/local_vroom \
 && chown -R app:app /var/tmp/local_vroom /app
ENV LOCAL_VROOM_TMPDIR=/var/tmp/local_vroom

USER app

# Le seul port ouvert du conteneur est celui de Gunicorn. VROOM n'en ouvre
# aucun : il lit un fichier, ecrit un fichier, et se termine.
EXPOSE 10000

# Commande de demarrage identique a celle du service Render actuel :
#   gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --access-logfile -
# Les workers et threads sont rendus explicites (au lieu des defauts implicites
# de Gunicorn) parce que le verrou d'optimisation et l'enveloppe memoire en
# dependent. Les defauts ci-dessous reproduisent le comportement actuel.
CMD ["sh", "-c", "exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-10000} \
    --workers ${GUNICORN_WORKERS:-1} \
    --threads ${GUNICORN_THREADS:-1} \
    --timeout ${GUNICORN_TIMEOUT:-300} \
    --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT:-30} \
    --access-logfile -"]
