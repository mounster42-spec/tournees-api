"""Concordance entre code.js (Google Sheets) et les diagnostics du backend.

Apps Script ne se teste pas depuis ce depot : aucun interpreteur JavaScript
n'est requis pour travailler ici. Ce qui PEUT etre verifie mecaniquement, et
qui casse le plus facilement en silence, c'est l'alignement :

  - un en-tete de colonne sans valeur correspondante decale TOUTE la ligne
    vers la gauche, et chaque cellule se retrouve sous le mauvais titre ;
  - une cle lue cote Sheets mais jamais produite par le backend remplit une
    colonne de vide sans que rien ne le signale.

Ces tests lisent donc code.js comme du texte et le confrontent aux
diagnostics reellement produits par app.py.
"""

import os
import re
import sys
import types
import unittest

if "flask" not in sys.modules:
    def _stub(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    class _FakeFlask:
        def __init__(self, *a, **k):
            pass

        def route(self, *a, **k):
            return lambda fn: fn

    _stub("flask", Flask=_FakeFlask, request=None, jsonify=lambda *a, **k: None)
    _stub("requests", post=None, get=None)
    _stub("numpy")
    sys.modules["sklearn"] = types.ModuleType("sklearn")
    _stub("sklearn.cluster", KMeans=object)
    _stub("sklearn.metrics", silhouette_score=None)

import app                                                        # noqa: E402
import local_vroom                                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_JS = os.path.join(HERE, "code.js")


def read_code():
    with open(CODE_JS, encoding="utf-8") as handle:
        return handle.read()


def strip_comments(text):
    """Retire les commentaires // et /* */ sans toucher aux chaines.

    Les chaines sont traitees EN PREMIER, et c'est indispensable : le `//`
    de "https://..." n'est pas un commentaire. Une fois dans un commentaire
    on saute jusqu'au bout sans regarder les guillemets, donc l'apostrophe
    d'un « n'est pas » ne peut pas ouvrir de fausse chaine non plus.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == "\\":
                    if i + 1 < n:
                        out.append(text[i + 1])
                        i += 2
                        continue
                elif text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def extract_block(text, opener):
    """Contenu entre les crochets qui suivent `opener`, profondeur respectee."""
    start = text.index(opener) + len(opener)
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] in "[({":
            depth += 1
        elif text[i] in "])}":
            depth -= 1
        i += 1
    return text[start:i - 1]


def extract_function(text, name):
    """Corps d'une fonction de premier niveau, sur le TEXTE BRUT.

    Volontairement sans analyse lexicale. Compter les accolades demande de
    reconnaitre chaines, echappements, litteraux d'expression reguliere et
    commentaires : un mini-analyseur JavaScript ecrit pour des tests se
    trompe plus souvent que le code qu'il surveille -- il s'est trompe ici
    meme, sur `!== "{"` puis sur le `//` de "https://".

    Toutes les fonctions de code.js et du gabarit sont formatees pareil :
    elles se terminent par une accolade en colonne zero. C'est ce reperage,
    exact et verifiable a l'oeil, qui est utilise.
    """
    marker = "\nfunction %s(" % name
    start = text.index(marker) + 1
    end = text.index("\n}\n", start)
    return text[start:end]


def split_top_level(block):
    """Decoupe sur les virgules de PROFONDEUR ZERO uniquement.

    Un simple split(",") couperait au milieu de _cell(a, b) et fausserait le
    comptage, ce qui est precisement l'erreur que ces tests cherchent.
    """
    items = []
    depth = 0
    current = []
    i = 0
    while i < len(block):
        ch = block[i]
        if ch in "\"'":
            quote = ch
            current.append(ch)
            i += 1
            while i < len(block):
                current.append(block[i])
                if block[i] == "\\" and i + 1 < len(block):
                    current.append(block[i + 1])
                    i += 2
                    continue
                if block[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return [item for item in items if item]


def header_list(name):
    code = strip_comments(read_code())
    block = extract_block(code, "const %s = [" % name)
    return [re.sub(r'^"|"$', "", item) for item in split_top_level(block)]


def row_expressions():
    code = strip_comments(read_code())
    block = extract_block(code, "const row = [")
    return split_top_level(block)


# Correspondance attendue, en ordre, entre les colonnes experimentales et
# l'expression qui les remplit. Une colonne derivee (calculee cote Sheets)
# est declaree ici explicitement : c'est le seul endroit ou l'ecart entre le
# nom de colonne et le nom de champ backend est autorise, et il est visible.
HYBRID_EXPECTED = [
    ("hybrid_error", "hyb.hybrid_error"),
    ("local_vroom_enabled", "hyb.local_vroom_enabled"),
    ("local_vroom_version", "hyb.local_vroom_version"),
    ("common_rescore_duration_s", "hyb.common_rescore_duration_s"),
    ("common_rescore_distance_m", "hyb.common_rescore_distance_m"),
    ("common_rescore_matrix_hash", "hyb.common_rescore_matrix_hash"),
    ("joint_direct_valid", "hyb.joint_direct_valid"),
    ("joint_direct_duration_s", "hyb.joint_direct_duration_s"),
    ("joint_direct_sizes", "hyb.joint_direct_sizes"),
    ("joint_nucleus_attempted", "hyb.joint_nucleus_attempted"),
    ("joint_nucleus_valid", "hyb.joint_nucleus_valid"),
    ("joint_nucleus_best_duration_s", "hyb.joint_nucleus_best_duration_s"),
    ("route_first_unique", "hyb.route_first_unique"),
    ("route_first_best_duration_s", "hyb.route_first_best_duration_s"),
    ("joint_alns_iterations", "hyb.joint_alns_iterations"),
    ("joint_alns_accepted", "hyb.joint_alns_accepted"),
    ("joint_alns_seed", "hyb.joint_alns_seed"),
    ("joint_alns_best_duration_s", "hyb.joint_alns_best_duration_s"),
    ("joint_finalists", "hyb.joint_finalists"),
    ("joint_finalists_local_vroom_solved",
     "hyb.joint_finalists_local_vroom_solved"),
    ("joint_finalists_reused", "hyb.joint_finalists_reused"),
    ("joint_solutions_considered", "hyb.joint_solutions_considered"),
    ("joint_selected_source", "hyb.joint_selected_source"),
    ("joint_selected_duration_s", "hyb.joint_selected_duration_s"),
    ("joint_selected_distance_m", "hyb.joint_selected_distance_m"),
    ("joint_selected_sizes", "hyb.joint_selected_sizes"),
    ("joint_selected_components", "hyb.joint_selected_components"),
    ("joint_selected_enclaves", "hyb.joint_selected_enclaves"),
    ("joint_territorial_level", "hyb.joint_territorial_level"),
    ("joint_territorial_max_enclaves", "hyb.joint_territorial_max_enclaves"),
    ("joint_territorial_admissible", "hyb.joint_territorial_admissible"),
    ("joint_territorial_fallback_used", "hyb.joint_territorial_fallback_used"),
    ("joint_territorial_fallback_reason",
     "hyb.joint_territorial_fallback_reason"),
    ("joint_territorial_thresholds", "hyb.joint_territorial_thresholds"),
    ("joint_territorial_level_counts", "hyb.joint_territorial_level_counts"),
    # Le plafond configure. La colonne porte desormais le nom exact du champ
    # backend : plus aucun ecart de nommage a retenir de ce cote.
    ("local_vroom_max_solves", "hyb.local_vroom_max_solves"),
    ("local_vroom_attempted", "hyb.local_vroom_attempted"),
    ("local_vroom_succeeded", "hyb.local_vroom_succeeded"),
    ("local_vroom_failed", "hyb.local_vroom_failed"),
    ("local_vroom_timed_out", "hyb.local_vroom_timed_out"),
    ("local_vroom_reused", "hyb.local_vroom_reused"),
    ("local_vroom_skipped_for_time", "hyb.local_vroom_skipped_for_time"),
    ("local_vroom_elapsed_ms", "hyb.local_vroom_elapsed_ms"),
    ("local_vroom_stop_reason", "hyb.local_vroom_stop_reason"),
    ("local_vroom_last_error", "hyb.local_vroom_last_error"),
    # Colonnes DERIVEES : un temps par bloc, extrait de hybrid_stages.
    ("hybrid_stage_matrix_ms", '_stageMs(stages, "matrix")'),
    ("hybrid_stage_joint_direct_ms", '_stageMs(stages, "joint_direct")'),
    ("hybrid_stage_route_first_ms", '_stageMs(stages, "route_first")'),
    ("hybrid_stage_joint_nucleus_ms", '_stageMs(stages, "joint_nucleus")'),
    ("hybrid_stage_joint_alns_ms", '_stageMs(stages, "joint_alns")'),
    ("hybrid_stage_alns_refine_ms", '_stageMs(stages, "alns_refine")'),
    ("hybrid_stage_joint_finalists_ms", '_stageMs(stages, "joint_finalists")'),
    ("hybrid_stage_timings_text", '_stagesText(stages, "elapsed_ms")'),
    ("hybrid_stage_stops_text", '_stagesText(stages, "stop_reason")'),
    ("hybrid_total_elapsed_ms", "hyb.total_elapsed_ms"),
    ("hybrid_soft_limit_reached", "hyb.soft_limit_reached"),
]


def backend_diagnostic_keys():
    """Toutes les cles que la strategie peut poser dans meta["hybrid"].

    Le diagnostic final est l'union du gabarit, du ledger, de la config et de
    l'horloge : c'est cette union que Sheets a le droit de lire.
    """
    keys = set(app._hybrid_diagnostics_template())
    keys |= set(local_vroom.LocalVroomLedger(max_solves=4).as_diagnostics())
    keys |= set(local_vroom.diagnostics(None, local_vroom.LocalVroomConfig()))
    keys |= set(app._HybridClock(58.0).as_diagnostics())
    return keys


# =========================================================================
# CONCORDANCE DES COLONNES
# =========================================================================

class TestBenchmarkColumns(unittest.TestCase):

    def setUp(self):
        self.headers = (header_list("BENCH_HEADERS_BASE")
                        + header_list("BENCH_HEADERS_D3")
                        + header_list("BENCH_HEADERS_TERR")
                        + header_list("BENCH_HEADERS_CONN")
                        + header_list("BENCH_HEADERS_ORSFIRST")
                        + header_list("BENCH_HEADERS_HYBRID"))
        self.row = row_expressions()

    def test_one_value_per_header(self):
        """Le test central : un en-tete de plus que de valeurs decale toute
        la ligne, et chaque cellule passe sous le mauvais titre."""
        self.assertEqual(len(self.headers), len(self.row),
                         "%d en-tetes pour %d valeurs"
                         % (len(self.headers), len(self.row)))

    def test_bench_headers_is_the_concatenation_in_order(self):
        code = strip_comments(read_code())
        concat = extract_block(code, "const BENCH_HEADERS = BENCH_HEADERS_BASE")
        for name in ("BENCH_HEADERS_D3", "BENCH_HEADERS_TERR",
                     "BENCH_HEADERS_CONN", "BENCH_HEADERS_ORSFIRST",
                     "BENCH_HEADERS_HYBRID"):
            self.assertIn(name, concat)
        # L'ordre de concatenation est celui des colonnes de la feuille.
        positions = [concat.index(name) for name in
                     ("BENCH_HEADERS_D3", "BENCH_HEADERS_TERR",
                      "BENCH_HEADERS_CONN", "BENCH_HEADERS_ORSFIRST",
                      "BENCH_HEADERS_HYBRID")]
        self.assertEqual(positions, sorted(positions))

    def test_no_duplicate_column_name(self):
        seen = {}
        for index, name in enumerate(self.headers):
            self.assertNotIn(name, seen,
                             "colonne %r en double (positions %s et %s)"
                             % (name, seen.get(name), index))
            seen[name] = index

    def test_existing_columns_are_untouched(self):
        """Les 99 colonnes historiques gardent leur nom ET leur rang."""
        historical = [
            "Date", "Stratégie exécutée", "Stratégie demandée", "Nb pts",
            "Signature jeu", "Nb véh", "Km T1", "Km T2", "Km total",
            "Min T1", "Min T2", "Min total", "Temps calcul (s)", "Appels API",
            "Vroom", "Matrix", "optimization_path", "Répartition",
        ]
        self.assertEqual(self.headers[:len(historical)], historical)
        # Derniere colonne d'avant l'ajout : elle ne bouge pas.
        orsfirst = header_list("BENCH_HEADERS_ORSFIRST")
        self.assertEqual(orsfirst[-1], "post_optimization_note")
        boundary = len(self.headers) - len(header_list("BENCH_HEADERS_HYBRID"))
        self.assertEqual(self.headers[boundary - 1], "post_optimization_note")

    def test_new_columns_are_strictly_appended(self):
        hybrid = header_list("BENCH_HEADERS_HYBRID")
        self.assertEqual(self.headers[-len(hybrid):], hybrid)
        # Aucune colonne experimentale ne s'intercale avant la fin.
        for name in hybrid:
            self.assertGreaterEqual(self.headers.index(name),
                                    len(self.headers) - len(hybrid))

    def test_each_experimental_column_reads_the_declared_field(self):
        hybrid = header_list("BENCH_HEADERS_HYBRID")
        tail = self.row[-len(hybrid):]
        self.assertEqual(len(HYBRID_EXPECTED), len(hybrid))
        for index, (name, expression) in enumerate(HYBRID_EXPECTED):
            self.assertEqual(hybrid[index], name,
                             "colonne %d attendue %r, trouvee %r"
                             % (index, name, hybrid[index]))
            self.assertIn(expression, tail[index],
                          "la colonne %r ne lit pas %r mais %r"
                          % (name, expression, tail[index]))


# =========================================================================
# CONCORDANCE AVEC LE BACKEND
# =========================================================================

class TestBackendFields(unittest.TestCase):

    def test_every_field_read_by_sheets_exists_in_the_backend(self):
        """Une colonne qui lit une cle inexistante reste vide sans alerter."""
        available = backend_diagnostic_keys()
        for name, expression in HYBRID_EXPECTED:
            if not expression.startswith("hyb."):
                continue
            field = expression[len("hyb."):]
            self.assertIn(field, available,
                          "la colonne %r lit hybrid.%s, que le backend ne "
                          "produit jamais" % (name, field))

    def test_the_requested_diagnostics_are_all_present(self):
        """La liste explicitement demandee, colonne par colonne."""
        headers = header_list("BENCH_HEADERS_HYBRID")
        for name in ("joint_selected_source", "joint_selected_duration_s",
                     "joint_selected_distance_m", "joint_selected_enclaves",
                     "joint_territorial_level", "joint_territorial_max_enclaves",
                     "joint_territorial_admissible",
                     "joint_territorial_fallback_used",
                     "joint_territorial_fallback_reason",
                     "joint_territorial_thresholds",
                     "joint_territorial_level_counts",
                     "joint_direct_duration_s", "joint_nucleus_best_duration_s",
                     "route_first_best_duration_s", "joint_alns_best_duration_s",
                     "local_vroom_attempted", "local_vroom_succeeded",
                     "local_vroom_elapsed_ms", "local_vroom_max_solves",
                     "common_rescore_duration_s", "common_rescore_distance_m"):
            self.assertIn(name, headers)

    def test_a_column_exists_for_every_stage_the_backend_declares(self):
        """Les blocs sont connus : un temps dedie existe pour chacun."""
        headers = header_list("BENCH_HEADERS_HYBRID")
        for stage in ("matrix", "joint_direct", "route_first", "joint_nucleus",
                      "joint_alns", "alns_refine", "joint_finalists"):
            self.assertIn("hybrid_stage_%s_ms" % stage, headers)
        # Et un texte de repli, pour qu'un bloc ajoute plus tard ne soit pas
        # perdu en silence.
        self.assertIn("hybrid_stage_timings_text", headers)

    def test_the_hybrid_block_travels_where_sheets_reads_it(self):
        """Sheets lit result.ors_matrix.hybrid ; le backend l'y place bien."""
        code = strip_comments(read_code())
        self.assertIn("meta.hybrid", code)
        self.assertIn("result.ors_matrix", code)
        with open(os.path.join(HERE, "app.py"), encoding="utf-8") as handle:
            backend = handle.read()
        self.assertIn('meta = {"hybrid": diag}', backend)
        self.assertIn('"ors_matrix": matrix_meta', backend)


# =========================================================================
# MENU ET STRATEGIE
# =========================================================================

class TestMenuWiring(unittest.TestCase):

    def setUp(self):
        self.code = strip_comments(read_code())

    def test_the_main_entry_is_optimiser_les_tournees(self):
        self.assertIn('const MENU_OPTIMISER_LABEL = "Optimiser les tournées";',
                      self.code)
        self.assertIn(".addItem(MENU_OPTIMISER_LABEL, "
                      '"runHybridLocalVroomTerritorial")', self.code)

    def test_the_main_entry_maps_to_the_hybrid_strategy(self):
        self.assertIn('const EXP_STRATEGY = "hybrid_local_vroom_territorial";',
                      self.code)
        self.assertIn("function runHybridLocalVroomTerritorial() "
                      "{ runOptimisation(EXP_STRATEGY); }", self.code)

    def test_no_exp_prefix_is_visible_anywhere(self):
        """Le marqueur technique ne doit plus apparaitre a l'utilisateur, ni
        dans le menu ni dans le champ Mode de la feuille Resultats."""
        self.assertNotIn("[EXP]", read_code())

    def test_the_strategy_id_matches_the_backend_exactly(self):
        match = re.search(r'const EXP_STRATEGY = "([^"]+)";', self.code)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "hybrid_local_vroom_territorial")
        self.assertIn(match.group(1), app.VALID_STRATEGIES)

    def test_the_strategy_is_accepted_by_the_parameters_sheet(self):
        strategies = header_list("STRATEGIES")
        self.assertIn("hybrid_local_vroom_territorial", strategies)

    def test_production_strategies_keep_their_order(self):
        strategies = header_list("STRATEGIES")
        self.assertEqual(tuple(strategies[:4]), app.PRODUCTION_STRATEGIES)
        self.assertEqual(strategies[4], "hybrid_local_vroom_territorial")

    def test_the_default_strategy_is_unchanged(self):
        self.assertIn('const DEFAULT_STRATEGY = "kmeans";', self.code)

    def test_the_menu_has_exactly_the_expected_entries_in_order(self):
        """Ordre exact du menu, entree par entree.

        L'ordre porte l'intention : l'usage quotidien d'abord, les methodes de
        comparaison releguees en dernier."""
        flat = re.sub(r"\s+", " ", self.code)
        menu = flat[flat.index('ui.createMenu("Tournées")'):]
        menu = menu[:menu.index(".addToUi()")]
        found = re.findall(r'\.addItem\(([^,]+), "(\w+)"\)|\.addSeparator\(\)'
                           r'|ui\.createMenu\("([^"]+)"\)', menu)
        sequence = []
        for label, handler, submenu in found:
            if submenu:
                sequence.append(("sous-menu", submenu))
            elif label:
                sequence.append((label.strip().strip('"'), handler))
            else:
                sequence.append(("separateur", ""))
        self.assertEqual(sequence, [
            ("sous-menu", "Tournées"),
            ("MENU_OPTIMISER_LABEL", "runHybridLocalVroomTerritorial"),
            ("Afficher la dernière carte", "afficherDerniereCarte"),
            ("Exporter la dernière carte", "exporterCartePartageable"),
            ("Ouvrir le benchmark", "ouvrirBenchmark"),
            ("separateur", ""),
            ("sous-menu", "Méthodes de comparaison"),
            ("K-means — référence", "runKmeans"),
            ("ORS connecté — comparaison", "runOrtoolsOrsMatrixConnected"),
        ])

    def test_the_export_entry_says_export_because_it_writes_to_drive(self):
        """Le libellé doit décrire le comportement réel : l'action crée un
        fichier Drive avant de proposer un lien, elle ne télécharge pas
        directement."""
        self.assertIn("DriveApp.createFile", self.code)
        flat = re.sub(r"\s+", " ", self.code)
        self.assertIn('.addItem("Exporter la dernière carte", '
                      '"exporterCartePartageable")', flat)

    def test_the_comparison_submenu_holds_exactly_two_methods(self):
        flat = re.sub(r"\s+", " ", self.code)
        submenu = flat[flat.index('ui.createMenu("Méthodes de comparaison")'):]
        submenu = submenu[:submenu.index(".addToUi()")]
        self.assertIn('.addItem("K-means — référence", "runKmeans")', submenu)
        self.assertIn('.addItem("ORS connecté — comparaison", '
                      '"runOrtoolsOrsMatrixConnected")', submenu)
        self.assertEqual(submenu.count(".addItem("), 2)

    def test_technical_methods_left_the_visible_menu_but_keep_their_code(self):
        """Retirees du menu, jamais supprimees : le backend les accepte encore
        et elles restent executables depuis l'editeur Apps Script."""
        flat = re.sub(r"\s+", " ", self.code)
        menu = flat[flat.index('ui.createMenu("Tournées")'):]
        menu = menu[:menu.index(".addToUi()")]
        # Comparaison sur la liste EXACTE des gestionnaires : "runOrtoolsOrsMatrix"
        # est un prefixe de "runOrtoolsOrsMatrixConnected", une recherche par
        # sous-chaine conclurait a tort que le premier est encore au menu.
        wired = set(re.findall(r'\.addItem\([^,]+, "(\w+)"\)', menu))
        for handler in ("runOrtoolsHaversine", "runOrtoolsOrsMatrix",
                        "clearResults", "resetSelection", "runOptimisation"):
            self.assertNotIn(handler, wired,
                             "%s ne devrait plus figurer au menu" % handler)
            self.assertIn("function %s(" % handler, self.code,
                          "%s a ete supprime alors qu'il doit rester" % handler)

    def test_every_menu_handler_exists(self):
        flat = re.sub(r"\s+", " ", self.code)
        menu = flat[flat.index('ui.createMenu("Tournées")'):]
        menu = menu[:menu.index(".addToUi()")]
        for handler in re.findall(r'\.addItem\([^,]+, "(\w+)"\)', menu):
            self.assertIn("function %s(" % handler, self.code,
                          "l'entree de menu appelle %s, qui n'existe pas"
                          % handler)

    def test_all_backend_strategies_remain_declared(self):
        """Aucune strategie n'est retiree du backend ni de la validation."""
        strategies = header_list("STRATEGIES")
        for name in app.VALID_STRATEGIES:
            self.assertIn(name, strategies)
        self.assertEqual(len(app.VALID_STRATEGIES), 5)

    def test_the_experimental_strategy_is_labelled_in_the_result_sheet(self):
        """Le champ Mode de la feuille Resultats nomme la strategie, sans
        marqueur technique."""
        self.assertIn("VROOM local conjoint + ALNS territoriale", self.code)
        self.assertNotIn("[EXP] VROOM local conjoint", self.code)


# =========================================================================
# INTEGRITE DU FICHIER
# =========================================================================

class TestCodeJsIntegrity(unittest.TestCase):

    def test_the_edited_regions_are_balanced(self):
        """Aucun interpreteur JavaScript n'est disponible dans ce depot.

        Un controle d'equilibrage sur le fichier ENTIER n'est pas fiable :
        code.js contient des litteraux HTML et des expressions que ce simple
        balayage lit de travers, avant comme apres cette modification. Le
        controle porte donc sur les regions reellement editees, ou il est
        exact : si un crochet y manquait, l'extraction ci-dessous echouerait
        ou rendrait un nombre d'elements faux.
        """
        code = strip_comments(read_code())
        for name in ("BENCH_HEADERS_HYBRID", "STRATEGIES"):
            items = header_list(name)
            self.assertTrue(items, "%s vide ou illisible" % name)
            for item in items:
                self.assertNotIn("[", item)
                self.assertNotIn("]", item)

        row = row_expressions()
        self.assertTrue(row)
        for expression in row:
            self.assertEqual(expression.count("("), expression.count(")"),
                             "parentheses desequilibrees dans %r" % expression)

        for helper in ("_hybridDiag", "_stageMs", "_stagesText", "_cellCounts"):
            self.assertTrue(extract_function(read_code(), helper).strip())

    def test_helpers_used_by_the_row_are_defined(self):
        code = strip_comments(read_code())
        for helper in ("_hybridDiag", "_stageMs", "_stagesText", "_cellCounts"):
            self.assertIn("function %s(" % helper, code,
                          "%s est utilise sans etre defini" % helper)

    def test_collecte_gs_is_gone_and_leaves_no_orphan_reference(self):
        """collecte.gs etait une implementation ANTERIEURE et autonome.

        Elle visait une autre structure de classeur -- feuille 'Parametres'
        sans accent, sorties 'Tournee_J1'/'Tournee_J2' -- appelait ORS
        directement avec une cle stockee cote script, et declarait son propre
        onOpen, en collision avec celui de code.js. Aucun de ses symboles
        n'etait appele par code.js ni par les fichiers HTML."""
        self.assertFalse(os.path.exists(os.path.join(HERE, "collecte.gs")),
                         "collecte.gs est de retour dans le depot")

        removed = ("calculerTournee", "configurerCleORS", "effacerTournee",
                   "initialiserClasseur", "reinitialiserSelection")
        sources = [os.path.join(HERE, name) for name in
                   ("code.js", "carte_tournee.html", "app.py",
                    "local_vroom.py")]
        for path in sources:
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as handle:
                content = handle.read()
            for symbol in removed:
                self.assertNotIn(symbol, content,
                                 "%s reference encore %s" % (path, symbol))

    def test_only_one_onopen_remains(self):
        """Deux fichiers .gs partagent le meme espace global : deux onOpen se
        masquaient l'un l'autre, et un seul menu apparaissait."""
        code = strip_comments(read_code())
        self.assertEqual(len(re.findall(r"^function onOpen\(", code, re.M)), 1)

    def test_the_benchmark_columns_needed_no_flask_route(self):
        """Les 155 colonnes se lisent dans la reponse existante.

        /map-geometry sert uniquement au TRACE de la carte : il n'alimente
        aucune colonne du Benchmark et n'est jamais appele par /optimize.
        La liste des routes est verifiee dans tests_map_geometry.py."""
        with open(os.path.join(HERE, "app.py"), encoding="utf-8") as handle:
            backend = handle.read()
        routes = sorted(re.findall(r'@app\.route\("([^"]+)"', backend))
        self.assertEqual(routes, ["/", "/healthz", "/map-geometry", "/optimize"])
        # Aucune colonne du Benchmark ne provient de la geometrie.
        for header in header_list("BENCH_HEADERS_HYBRID"):
            self.assertNotIn("geometry", header)
            self.assertNotIn("map_", header)


# =========================================================================
# CARTE : TRACE ROUTIER ET TELECHARGEMENT
# =========================================================================

def read_map_html():
    with open(os.path.join(HERE, "carte_tournee.html"), encoding="utf-8") as h:
        return h.read()


class TestMapDownloadButton(unittest.TestCase):

    def setUp(self):
        self.html = read_map_html()
        self.code = strip_comments(read_code())

    def test_the_button_exists_and_is_wired(self):
        self.assertIn('id="btn-download"', self.html)
        self.assertIn("Télécharger la carte", self.html)
        self.assertIn('getElementById("btn-download")', self.html)
        self.assertIn("telechargerCarte", self.html)

    def test_the_button_is_hidden_outside_apps_script(self):
        """Un fichier exporte n'a ni serveur ni bouton : tout y est deja."""
        self.assertIn('id="actions" style="display:none"', self.html)
        self.assertIn('getElementById("actions").style.display = "block"',
                      self.html)

    def test_the_download_goes_through_drive_not_a_blob(self):
        """Le telechargement direct depuis une iframe HtmlService n'est pas
        garanti : elle est sandboxee. On passe par Drive puis un vrai lien."""
        self.assertIn("exporterCarteDepuisDialogue", self.html)
        self.assertNotIn("createObjectURL", self.html)
        self.assertNotIn("URL.createObjectURL", self.code)
        self.assertIn("function exporterCarteDepuisDialogue(", self.code)

    def test_the_export_reuses_the_single_standalone_builder(self):
        """Aucune seconde logique d'export : les deux chemins passent par le
        meme constructeur."""
        self.assertEqual(self.code.count("function _buildStandaloneCarteHtml_("), 1)
        self.assertEqual(self.code.count("_buildStandaloneCarteHtml_("), 3)
        self.assertEqual(self.code.count("DriveApp.createFile("), 1)

    def test_the_filename_is_deterministic_and_sanitised(self):
        self.assertIn("function _nomFichierCarte_(", self.code)
        self.assertIn('"carte_tournees_" + propre + "_" + stamp + ".html"',
                      self.code)
        self.assertIn('replace(/[^A-Za-z0-9_-]/g, "")', self.code)
        self.assertIn('"yyyy-MM-dd_HH-mm"', self.code)

    def test_exporting_launches_no_optimisation(self):
        export = extract_function(read_code(), "exporterCarteDepuisDialogue")
        for forbidden in ("callAPI", "runOptimisation", "appendBenchmark",
                          "getPoints("):
            self.assertNotIn(forbidden, export)

    def test_exporting_never_writes_the_geometry_back_to_the_sheet(self):
        """Une cellule plafonne a 50 000 caracteres : deux traces routiers la
        feraient sauter. La geometrie ne vit que dans la fenetre et le
        fichier."""
        export = extract_function(read_code(), "exporterCarteDepuisDialogue")
        self.assertNotIn("_saveCartePayload_", export)
        self.assertNotIn("setValue", export)
        merge = extract_function(read_code(), "_payloadAvecGeometries_")
        self.assertNotIn("_saveCartePayload_", merge)
        self.assertNotIn("setValue", merge)
        # Le seul ecrivain de _CarteData reste la sauvegarde du run.
        self.assertEqual(self.code.count("function _saveCartePayload_("), 1)
        self.assertEqual(self.code.count("_saveCartePayload_("), 2)

    def test_the_window_export_sends_the_geometry_it_already_has(self):
        """Deja chargee, la geometrie repart telle quelle : l'export ne
        redemande jamais le trace et ne coute aucun appel de plus."""
        self.assertIn("currentGeometries ? JSON.stringify(currentGeometries)",
                      self.html)
        export = extract_function(read_code(), "exporterCarteDepuisDialogue")
        self.assertNotIn("getCarteGeometrie", export)
        self.assertNotIn("UrlFetchApp", export)


class TestMapGeometryWiring(unittest.TestCase):

    def setUp(self):
        self.html = read_map_html()
        self.code = strip_comments(read_code())

    def test_the_map_is_rendered_before_the_geometry_is_requested(self):
        boot = self.html[self.html.index("function boot()"):]
        self.assertLess(boot.index("renderCarte(data)"),
                        boot.index("chargerGeometrie()"))

    def test_a_loading_state_is_shown(self):
        self.assertIn("Chargement du tracé routier", self.html)
        self.assertIn("Tracé routier chargé", self.html)

    def test_the_fallback_message_is_explicit(self):
        self.assertIn("Tracé indicatif", self.html)
        self.assertIn("serveur injoignable", self.html)

    def test_geometry_draws_a_solid_line_and_its_absence_a_dashed_one(self):
        draw = self.html[self.html.index("function drawRoute("):]
        draw = draw[:draw.index("\n}\n")]
        self.assertIn("route.geometry && route.geometry.length > 1", draw)
        self.assertIn('dashArray:"7,7"', draw)
        self.assertIn("tracé indicatif, non routier", draw)

    def test_the_point_order_is_never_recomputed(self):
        loader = extract_function(self.html, "chargerGeometrie")
        for forbidden in ("sort(", "reverse(", "splice("):
            self.assertNotIn(forbidden, loader)
        self.assertIn("route.push([lon, lat])", loader)

    def test_no_ors_key_anywhere_on_the_client_side(self):
        """La cle vit uniquement sur Render. C'est la raison d'etre de
        l'endpoint : sans lui, Apps Script devrait la porter."""
        for content in (read_code(), self.html):
            lowered = content.lower()
            for needle in ("ors_key", "ors_api_key", "api.openrouteservice.org",
                           "heigit.org", "authorization"):
                self.assertNotIn(needle, lowered)

    def test_the_client_only_talks_to_our_own_backend(self):
        """Une seule origine sortante dans tout le fichier : notre backend.

        Le controle porte sur les LITTERAUX d'URL. Une adresse tierce, meme
        introduite par megarde, apparaitrait ici."""
        self.assertIn('API_BASE + "/map-geometry"', self.code)
        literals = set(re.findall(r'"(https?://[^"]*)"', self.code))
        self.assertEqual(literals, {
            "https://tournees-api.onrender.com",
            # Lien AFFICHE a l'utilisateur, pas un appel sortant du script.
            "https://drive.google.com/uc?export=download&id=",
        })

    def test_the_geometry_request_carries_exactly_two_routes(self):
        loader = extract_function(read_code(), "getCarteGeometrie")
        self.assertIn("routes.length !== 2", loader)
        self.assertIn('profile: "driving-car"', loader)

    def test_the_geometry_call_never_wakes_the_server(self):
        """callAPI reveille Render jusqu'a une minute. La carte, elle, est
        deja affichee : elle ne doit pas faire patienter devant un demarrage
        a froid, elle retombe sur ses pointilles."""
        loader = extract_function(read_code(), "getCarteGeometrie")
        self.assertNotIn("Utilities.sleep", loader)
        self.assertNotIn("for (var attempt", loader)

    def test_the_geometry_helper_never_throws(self):
        loader = extract_function(read_code(), "getCarteGeometrie")
        self.assertIn("catch (e)", loader)
        self.assertIn("fallback_used: true", loader)

    def test_the_optimisation_path_never_requests_geometry(self):
        run = extract_function(read_code(), "runOptimisation")
        self.assertNotIn("getCarteGeometrie", run)
        self.assertNotIn("map-geometry", run)

    def test_the_benchmark_row_carries_no_geometry(self):
        for expression in row_expressions():
            self.assertNotIn("geometr", expression.lower())


if __name__ == "__main__":
    unittest.main()
