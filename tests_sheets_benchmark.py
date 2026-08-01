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

    def test_the_root_menu_is_named_menu_tournees(self):
        """« Menu tournees » se repere mieux parmi les menus de Sheets."""
        self.assertIn('const MENU_RACINE_LABEL = "Menu tournées";', self.code)
        self.assertIn("ui.createMenu(MENU_RACINE_LABEL)", self.code)
        # Le titre passe par la constante, jamais par une chaine recopiee.
        self.assertNotIn('ui.createMenu("Tournées")', self.code)
        self.assertNotIn('ui.createMenu("Menu tournées")', self.code)

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

        L'ordre porte l'intention : les trois gestes d'une journee au premier
        niveau, tout l'outillage technique sous « Outils dev »."""
        flat = re.sub(r"\s+", " ", self.code)
        menu = flat[flat.index('ui.createMenu(MENU_RACINE_LABEL)'):]
        menu = menu[:menu.index(".addToUi()")]
        # Le titre racine passe par une constante, les sous-menus par des
        # chaines : les deux formes doivent etre reconnues.
        found = re.findall(r'\.addItem\(([^,]+), "(\w+)"\)|\.addSeparator\(\)'
                           r'|ui\.createMenu\("?([^")]+)"?\)', menu)
        sequence = []
        for label, handler, submenu in found:
            if submenu:
                sequence.append(("sous-menu", submenu))
            elif label:
                sequence.append((label.strip().strip('"'), handler))
            else:
                sequence.append(("separateur", ""))
        self.assertEqual(sequence, [
            ("sous-menu", "MENU_RACINE_LABEL"),
            ("Sélectionner les points par ID", "ouvrirSelectionParId"),
            ("MENU_OPTIMISER_LABEL", "runHybridLocalVroomTerritorial"),
            ("Ouvrir la carte", "ouvrirLaCarte"),
            ("separateur", ""),
            ("sous-menu", "Outils dev"),
            ("Ouvrir le benchmark", "ouvrirBenchmark"),
            ("sous-menu", "Méthodes de comparaison"),
            ("K-means — référence", "runKmeans"),
            ("ORS connecté — comparaison", "runOrtoolsOrsMatrixConnected"),
        ])

    def test_the_old_duplicate_map_entries_are_gone(self):
        """Deux entrees carte se ressemblaient trop. Il n'en reste qu'une."""
        flat = re.sub(r"\s+", " ", self.code)
        self.assertNotIn('"Afficher la dernière carte"', flat)
        self.assertNotIn('"Ouvrir la dernière carte"', flat)
        self.assertEqual(flat.count('.addItem("Ouvrir la carte"'), 1)

    def test_the_export_entry_left_the_menu_but_keeps_its_function(self):
        """L'export existe DEJA dans la carte ouverte.

        Le proposer une seconde fois au menu laissait croire que l'archive
        Drive etait le moyen normal de consulter la carte. La fonction, elle,
        reste executable depuis l'editeur Apps Script."""
        flat = re.sub(r"\s+", " ", self.code)
        menu = flat[flat.index('ui.createMenu(MENU_RACINE_LABEL)'):]
        menu = menu[:menu.index(".addToUi()")]
        self.assertNotIn("Exporter la carte", menu)
        self.assertNotIn("exporterCartePartageable", menu)
        self.assertIn("function exporterCartePartageable(", self.code)
        # L'action d'export interne, elle, n'a pas bouge.
        self.assertIn("function exporterCarteDepuisDialogue(", self.code)
        self.assertIn("DriveApp.createFile", self.code)

    def test_the_developer_tools_submenu_gathers_the_technical_entries(self):
        """« Ouvrir le benchmark » et les methodes de comparaison n'encombrent
        plus le premier niveau."""
        flat = re.sub(r"\s+", " ", self.code)
        outils = flat[flat.index('ui.createMenu("Outils dev")'):]
        outils = outils[:outils.index(".addToUi()")]
        self.assertIn('.addItem("Ouvrir le benchmark", "ouvrirBenchmark")',
                      outils)
        self.assertIn('ui.createMenu("Méthodes de comparaison")', outils)
        # Le gestionnaire existant est reutilise tel quel, sans duplication.
        self.assertEqual(self.code.count("function ouvrirBenchmark("), 1)

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
        menu = flat[flat.index('ui.createMenu(MENU_RACINE_LABEL)'):]
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
        menu = flat[flat.index('ui.createMenu(MENU_RACINE_LABEL)'):]
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
# SELECTION PAR ID
# =========================================================================

def read_sidebar():
    with open(os.path.join(HERE, "selection_par_id.html"), encoding="utf-8") as h:
        return h.read()


class TestSelectionParId(unittest.TestCase):

    def setUp(self):
        self.code = read_code()
        self.html = read_sidebar()

    # --- lecture seule des depots ---------------------------------------

    def test_the_depots_are_read_but_never_written(self):
        """Paramètres!B4 et B5 restent pilotes depuis la feuille.

        La barre laterale les AFFICHE ; les corriger reste une action dans la
        feuille. Aucune ecriture, dans aucune des fonctions du bloc."""
        for name in ("getSelectionContexte", "appliquerSelectionParId",
                     "_lireHorodateurs_", "_resoudreId_", "_parseIds_"):
            body = extract_function(self.code, name)
            # setValues() groupe est autorise sur la colonne des cases ;
            # setValue() unitaire ne l'est nulle part, et c'est la seule
            # forme qui pourrait atteindre B4 ou B5.
            self.assertNotIn("setValue(", body,
                             "%s ecrit une cellule isolee" % name)

        # Les deux cellules sont lues, et uniquement lues.
        lecture = extract_function(self.code, "getSelectionContexte")
        self.assertIn("getRange(PARAM_ROW_START, 2).getValue()", lecture)
        self.assertIn("getRange(PARAM_ROW_END, 2).getValue()", lecture)

    def test_the_depot_fields_are_not_editable(self):
        """Aucun champ de saisie pour le depart ni pour l'arrivee."""
        self.assertIn('<div class="depot" id="depart">', self.html)
        self.assertIn('<div class="depot" id="arrivee">', self.html)
        # Le seul champ modifiable de la barre laterale est la zone de collage.
        champs = re.findall(r"<(input|textarea|select)\b", self.html)
        self.assertEqual(champs, ["textarea"])

    def test_the_sidebar_never_offers_to_fix_the_depots(self):
        for interdit in ("Corriger le départ", "Modifier Paramètres",
                         "setDepart", "setArrivee"):
            self.assertNotIn(interdit, self.html)
        self.assertIn("se modifient dans la feuille Paramètres", self.html)

    # --- perimetre d'ecriture -------------------------------------------

    def test_the_only_written_range_is_the_checkbox_column(self):
        """Une seule plage ecrite dans tout le bloc, et c'est la colonne E."""
        body = extract_function(self.code, "appliquerSelectionParId")
        ecritures = re.findall(r"\.setValues?\(", body)
        self.assertEqual(len(ecritures), 1,
                         "attendu une seule ecriture, trouve %d" % len(ecritures))
        self.assertIn("getRange(2, HORO_COL_SEL, valeurs.length, 1).setValues(",
                      body)
        self.assertIn("const HORO_COL_SEL = 5;", self.code)

    def test_no_coordinate_no_id_no_address_is_ever_written(self):
        """Les colonnes A a D ne sont jamais ecrites, sous aucune forme."""
        body = extract_function(self.code, "appliquerSelectionParId")
        for interdit in ("HORO_COL_ID,", "getRange(2, 1,", "getRange(2, 2,",
                         "getRange(2, 3,", "getRange(2, 4,", "setFormula",
                         "insertRows", "deleteRow", "moveRows", ".sort("):
            self.assertNotIn(interdit, body,
                             "appliquerSelectionParId touche %s" % interdit)

    def test_nothing_is_written_row_by_row(self):
        body = extract_function(self.code, "appliquerSelectionParId")
        self.assertNotIn("setValue(", body)
        self.assertNotIn("check()", body)
        self.assertNotIn("uncheck()", body)

    def test_a_blocking_depot_writes_nothing(self):
        """Chaque sortie bloquante rend le rapport AVANT l'ecriture."""
        body = extract_function(self.code, "appliquerSelectionParId")
        ecriture = body.index(".setValues(")
        for garde in ("if (!startId)", "if (!depart.trouve)",
                      "if (!arrivee.trouve)", "if (!parsed.uniques.length)",
                      "if (rapport.inconnus.length || rapport.ambigus.length)"):
            self.assertLess(body.index(garde), ecriture,
                            "la garde %s arrive apres l'ecriture" % garde)

    # --- identifiants ----------------------------------------------------

    def test_ids_are_matched_on_both_the_raw_and_displayed_form(self):
        """getValues rend 12 la ou l'utilisateur voit 0012 : les deux formes
        sont indexees, donc coller l'une ou l'autre fonctionne."""
        body = extract_function(self.code, "_lireHorodateurs_")
        self.assertIn("getDisplayValues()", body)
        self.assertIn("getValues()", body)
        self.assertIn("[ligne.idAffiche, ligne.idBrut]", body)

    def test_no_identifier_is_ever_converted(self):
        body = extract_function(self.code, "_parseIds_")
        for interdit in ("Number(", "parseInt", "parseFloat", "toUpperCase",
                         "toLowerCase", "replace(/^0+/"):
            self.assertNotIn(interdit, body,
                             "_parseIds_ transforme les identifiants: %s" % interdit)
        self.assertIn(".trim()", body)

    def test_the_accepted_separators_do_not_include_the_space(self):
        """Un identifiant peut contenir une espace : elle ne peut pas servir
        de separateur sans casser des ID reels."""
        body = extract_function(self.code, "_parseIds_")
        self.assertIn("split(/[\\n\\r,;\\t]+/)", body)
        self.assertNotIn("\\s+/", body)

    def test_duplicate_input_is_deduplicated_but_counted(self):
        body = extract_function(self.code, "_parseIds_")
        self.assertIn("saisis++", body)
        self.assertIn("if (vus[id]) continue;", body)

    def test_an_ambiguous_id_is_never_resolved_arbitrarily(self):
        resolve = extract_function(self.code, "_resoudreId_")
        self.assertIn("trouve: rows.length === 1", resolve)
        self.assertIn("ambigu: rows.length > 1", resolve)
        body = extract_function(self.code, "appliquerSelectionParId")
        self.assertIn("if (trouve.ambigu) { rapport.ambigus.push", body)

    def test_a_depot_typed_as_a_collection_is_reported_not_swallowed(self):
        body = extract_function(self.code, "appliquerSelectionParId")
        self.assertIn("rapport.depotsDansCollectes.push(id)", body)

    # --- cardinalite -----------------------------------------------------

    def test_the_same_depot_is_checked_once(self):
        """rowsDepots est un objet indexe par numero de ligne : depart et
        arrivee sur la meme ligne n'y entrent qu'une fois."""
        body = extract_function(self.code, "appliquerSelectionParId")
        self.assertIn("rowsDepots[depart.rows[0]] = true;", body)
        self.assertIn("rowsDepots[arrivee.rows[0]] = true;", body)
        self.assertIn("Object.keys(rowsDepots).length", body)

    def test_the_total_counts_collections_plus_depots(self):
        body = extract_function(self.code, "appliquerSelectionParId")
        self.assertIn("rapport.lignesCochees = rowsCollectes.length + nbDepots;",
                      body)

    def test_the_success_message_matches_the_requested_wording(self):
        self.assertIn("Départ et arrivée validés.", self.html)
        self.assertIn("points de collecte sélectionnés.", self.html)
        self.assertIn("Vous pouvez maintenant lancer l'optimisation.", self.html)

    def test_the_sidebar_never_starts_an_optimisation(self):
        for interdit in ("runOptimisation", "callAPI", "appendBenchmark"):
            self.assertNotIn(interdit, self.html)

    def test_the_sidebar_escapes_what_it_displays(self):
        """Les libelles viennent du Sheet et peuvent contenir n'importe quoi."""
        self.assertIn("function esc(", self.html)
        self.assertIn('replace(/</g,"&lt;")', self.html)

    def test_the_apply_button_is_touch_sized(self):
        self.assertIn("min-height:44px", self.html)


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
        # Le popup n'a plus que deux actions : « Ouvrir en grand » et
        # « Exporter ». L'ancien libelle « Télécharger la carte » disait mal
        # ce qui se passe -- le fichier part d'abord sur Drive.
        self.assertIn('id="btn-download"', self.html)
        self.assertIn(">Exporter et partager<", self.html)
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
        # Un constructeur, trois usages : l'archive de la fenetre, l'archive
        # depuis l'editeur, et la page servie par le lien de partage.
        self.assertEqual(self.code.count("_buildStandaloneCarteHtml_("), 4)
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


class TestWebApp(unittest.TestCase):
    """La Web App sert la meme page sur ordinateur et sur telephone, et ne
    montre rien a qui n'a pas acces au classeur."""

    def setUp(self):
        self.code = read_code()

    def test_doget_never_relies_on_an_active_spreadsheet(self):
        """Ni classeur actif, ni interface, ni feuille active : aucune de ces
        notions n'existe hors du classeur."""
        body = extract_function(self.code, "doGet")
        for interdit in ("getActive()", "getActiveSpreadsheet",
                         "getActiveSheet", "getUi()", "getActiveRange"):
            self.assertNotIn(interdit, body,
                             "doGet depend de %s" % interdit)

    def test_the_workbook_is_found_by_a_stored_identifier(self):
        self.assertIn('const PROP_SPREADSHEET_ID = "TOURNEES_SPREADSHEET_ID";',
                      self.code)
        # doGet ne connait plus le classeur : il sert un instantane designe
        # par un jeton. C'est _classeur_ qui porte l'ouverture par identifiant.
        body = extract_function(self.code, "_classeur_")
        self.assertIn("getProperty(PROP_SPREADSHEET_ID)", body)
        self.assertIn("SpreadsheetApp.openById(id)", body)
        # L'identifiant est enregistre depuis le classeur, a l'ouverture.
        self.assertIn("_memoriserClasseur_();",
                      extract_function(self.code, "onOpen"))

    def test_the_token_is_checked_before_anything_is_served(self):
        """Le jeton est la seule cle, et il est verifie avant toute lecture."""
        body = extract_function(self.code, "doGet")
        garde = body.index("if (!payload)")
        service = body.index("_buildStandaloneCarteHtml_(payload)")
        self.assertLess(garde, service,
                        "la page est assemblee avant le controle du jeton")
        self.assertLess(body.index("if (!jeton)"), garde)

    def test_an_unknown_or_expired_token_gets_one_single_answer(self):
        """Distinguer les cas revelerait qu'un jeton a existe."""
        # Sans commentaires : celui qui explique la regle doit la nommer.
        body = strip_comments(extract_function(self.code, "doGet"))
        self.assertEqual(body.count("MSG_PARTAGE_INDISPONIBLE"), 2)
        for interdit in ("expiré le", "jeton inconnu", "révoqué le"):
            self.assertNotIn(interdit, body)

    def test_the_refusal_page_carries_no_map_data(self):
        page = extract_function(self.code, "_pageMessage_")
        for interdit in ("getCarteTourneesPayload", "TOURNEES_PAYLOAD",
                         "routes", "geometr", "MAP_DATA_SHEET"):
            self.assertNotIn(interdit, page,
                             "la page de message expose %s" % interdit)

    def test_the_payload_reader_leaks_nothing_without_access(self):
        body = extract_function(self.code, "getCarteTourneesPayload")
        self.assertIn("catch (e)", body)
        self.assertIn("return null;", body)

    def test_the_workbook_helper_falls_back_to_open_by_id(self):
        body = extract_function(self.code, "_classeur_")
        self.assertIn("SpreadsheetApp.getActive()", body)
        self.assertIn("SpreadsheetApp.openById(id)", body)

    def test_a_single_responsive_page_serves_every_device(self):
        """Aucune version mobile separee : le meme gabarit partout."""
        body = extract_function(self.code, "doGet")
        self.assertIn("viewport-fit=cover", body)
        # Le dialogue lit le gabarit, l'export autonome aussi. La Web App sert
        # le fichier DEJA construit par l'export : le lien et le fichier
        # telecharge montrent donc exactement la meme page, et il n'existe
        # aucune version mobile a maintenir a part.
        self.assertEqual(
            self.code.count("createHtmlOutputFromFile(MAP_HTML_FILE)"), 2)
        self.assertEqual(self.code.count("const MAP_HTML_FILE"), 1)
        self.assertEqual(
            self.code.count("function _buildStandaloneCarteHtml_("), 1)

    def test_the_web_app_page_is_served_with_its_payload_injected(self):
        """L'inverse de l'ancienne regle, et c'est le but.

        Servi SANS payload, le gabarit reclamait ses donnees par
        google.script.run — donc un compte autorise sur le classeur. Injecte,
        il rend seul : c'est ce qui rend la page ouvrable par un ami."""
        body = extract_function(self.code, "doGet")
        self.assertIn("_buildStandaloneCarteHtml_(payload)", body)
        # Sans commentaires : celui qui explique la regle doit la nommer.
        sans = strip_comments(body)
        self.assertNotIn("google.script.run", sans)
        self.assertNotIn("getCarteTourneesPayload", sans)

    def test_no_permission_is_ever_changed_by_the_code(self):
        for interdit in ("setSharing", "addEditor", "addViewer", "ANYONE",
                         "setAccess", "setOwner", "removeEditor"):
            self.assertNotIn(interdit, self.code,
                             "le code modifie un partage : %s" % interdit)
        # DriveApp.getFileById redevient interdit : plus AUCUNE lecture Drive
        # sur le chemin de service. C'etait le seul maillon Drive entre
        # l'adresse script.google.com et le contenu.
        sans = strip_comments(self.code)
        self.assertNotIn("DriveApp.getFileById", sans)
        self.assertNotIn("getDownloadUrl", sans)
        # Drive ne sert plus qu'a DEPOSER l'archive.
        self.assertEqual(sans.count("DriveApp."), 1)
        self.assertIn("DriveApp.createFile(", sans)

    def test_the_backend_is_untouched_by_the_web_app(self):
        with open(os.path.join(HERE, "app.py"), encoding="utf-8") as handle:
            backend = handle.read()
        routes = sorted(re.findall(r'@app\.route\("([^"]+)"', backend))
        self.assertEqual(routes, ["/", "/healthz", "/map-geometry", "/optimize"])


class TestWebAppUrlIsTheOnlySource(unittest.TestCase):
    """L'ouverture en grand passait une adresse NON VALIDEE a window.open.

    getWebAppUrl rendait telle quelle la chaine reçue de la plateforme :
    `return url ? String(url) : "";`. Une adresse de brouillon en /dev, une
    adresse de classeur ou une adresse Drive heritee du chemin d'export
    partaient donc directement dans l'onglet. Drive n'affiche pas les fichiers
    HTML — il repond « Impossible d'ouvrir le fichier pour le moment ».

    Un seul helper produit desormais cette adresse, et il la regarde."""

    def setUp(self):
        self.code = read_code()
        self.html = read_map_html()

    def test_the_single_source_of_truth_is_the_script_service(self):
        helper = extract_function(self.code, "_getWebAppUrl_")
        self.assertIn("ScriptApp.getService().getUrl()", helper)
        self.assertIn("_validerUrlWebApp_(", helper)
        self.assertIn("catch (e)", helper)
        # Une seule lecture de la plateforme dans tout le CODE : aucune autre
        # fonction ne peut fabriquer une adresse d'ouverture. Le comptage
        # porte sur le code sans commentaires, la documentation du helper
        # citant elle-meme l'appel.
        self.assertEqual(
            strip_comments(self.code).count("ScriptApp.getService().getUrl()"),
            1)
        self.assertEqual(self.code.count("function _getWebAppUrl_("), 1)

    def test_the_rpc_surface_only_delegates(self):
        """getWebAppUrl reste appelable par google.script.run, sans logique."""
        wrapper = extract_function(self.code, "getWebAppUrl")
        self.assertIn("return _getWebAppUrl_();", wrapper)
        self.assertNotIn("ScriptApp", wrapper)
        # L'ancienne forme, celle qui rendait la chaine sans la regarder.
        self.assertNotIn('return url ? String(url) : "";', self.code)

    def test_only_an_exec_deployment_url_is_accepted(self):
        valider = extract_function(self.code, "_validerUrlWebApp_")
        self.assertIn('const WEB_APP_URL_SUFFIXE = "/exec";', self.code)
        self.assertIn("WEB_APP_URL_SUFFIXE", valider)
        self.assertIn('url.indexOf("https://") !== 0', valider)

    def test_a_drive_url_is_refused_fragment_by_fragment(self):
        interdits = header_list("WEB_APP_URL_INTERDITS")
        for fragment in ("drive.google.com", "/file/d/", "uc?export=",
                         "/view", "/download"):
            self.assertIn(fragment, interdits,
                          "%s doit disqualifier une adresse d'ouverture"
                          % fragment)
        valider = extract_function(self.code, "_validerUrlWebApp_")
        self.assertIn("WEB_APP_URL_INTERDITS", valider)
        self.assertIn('return "";', valider)

    def test_the_helper_is_no_longer_wired_to_the_menu(self):
        """Le menu n'ouvre plus la Web App : il rouvre le dialogue.

        Le helper reste, valide, car le PARTAGE en a besoin — mais plus
        aucune ouverture de carte ne construit d'adresse."""
        menu = extract_function(self.code, "ouvrirLaCarte")
        for interdit in ("_getWebAppUrl_", "getWebAppUrl", "ScriptApp",
                         "window.open", "http", "/exec", "drive.google.com"):
            self.assertNotIn(interdit, menu,
                             "ouvrirLaCarte touche encore %s" % interdit)

    def test_the_drive_archive_url_never_reaches_an_opening_function(self):
        """L'archive naît dans _deposerCarteSurDrive_ et n'en sort que pour
        l'export. Aucune fonction d'ouverture ne la voit."""
        depot = extract_function(self.code, "_deposerCarteSurDrive_")
        self.assertIn("file.getUrl()", depot)
        self.assertIn("uc?export=download", depot)
        for name in ("ouvrirLaCarte", "_getWebAppUrl_", "_validerUrlWebApp_",
                     "getWebAppUrl"):
            body = extract_function(self.code, name)
            for interdit in ("file.getUrl()", "_deposerCarteSurDrive_",
                             "uc?export=download"):
                self.assertNotIn(interdit, body,
                                 "%s manipule l'adresse de l'archive" % name)

class TestReopenLastMapFromMenu(unittest.TestCase):
    """« Ouvrir la carte » rouvre la derniere carte, dans le MEME dialogue.

    L'entree de menu passait par la Web App : petite fenetre intermediaire,
    nouvel onglet, adresse a construire. Alors que la carte de fin
    d'optimisation, elle, s'affichait tres bien dans un showModalDialog. Les
    deux parcours convergent desormais sur un seul helper."""

    def setUp(self):
        self.code = read_code()

    def test_a_single_helper_shows_the_dialog(self):
        self.assertEqual(self.code.count("function _afficherDialogueCarte_("),
                         1)
        helper = extract_function(self.code, "_afficherDialogueCarte_")
        self.assertIn("createHtmlOutputFromFile(MAP_HTML_FILE)", helper)
        self.assertIn("showModalDialog(out, MAP_DIALOG_TITLE)", helper)
        self.assertIn("MAP_DIALOG_WIDTH", helper)
        self.assertIn("MAP_DIALOG_HEIGHT", helper)

    def test_both_paths_converge_on_that_helper(self):
        for name in ("afficherCarteTournees", "ouvrirLaCarte"):
            body = extract_function(self.code, name)
            self.assertIn("_afficherDialogueCarte_();", body,
                          "%s n'utilise pas le helper commun" % name)
        # Deux appels, et deux seulement : la definition est exclue.
        appels = re.findall(r"(?<!function )_afficherDialogueCarte_\(\)",
                            self.code)
        self.assertEqual(len(appels), 2)
        # Un seul showModalDialog pour la carte dans tout le fichier.
        self.assertEqual(self.code.count("createHtmlOutputFromFile(MAP_HTML_FILE)\n"
                                         "    .setWidth(MAP_DIALOG_WIDTH)"), 1)

    def test_the_menu_opens_no_tab_and_no_web_app(self):
        body = extract_function(self.code, "ouvrirLaCarte")
        for interdit in ("window.open", "target=\"_blank\"", "/exec", "/dev",
                         "drive.google.com", "getWebAppUrl", "ScriptApp",
                         "HtmlService.createHtmlOutput("):
            self.assertNotIn(interdit, body,
                             "l'entree de menu passe encore par %s" % interdit)

    def test_no_payload_says_so_and_opens_nothing(self):
        body = extract_function(self.code, "ouvrirLaCarte")
        self.assertIn("if (!getCarteTourneesPayload())", body)
        self.assertIn("MSG_AUCUNE_CARTE", body)
        self.assertIn("return;", body)
        self.assertIn('const MSG_AUCUNE_CARTE =\n  "Aucune carte disponible. '
                      'Lancez d\'abord une optimisation.";', self.code)
        # La garde precede l'ouverture : jamais de dialogue vide.
        self.assertLess(body.index("if (!getCarteTourneesPayload())"),
                        body.index("_afficherDialogueCarte_();"))

    def test_the_duplicate_reopen_function_is_gone(self):
        """afficherDerniereCarte faisait exactement cela, hors du menu."""
        self.assertNotIn("afficherDerniereCarte", self.code)
        self.assertNotIn("_ouvrirDialogueCarte_", self.code)

    def test_reopening_launches_no_optimisation_and_writes_nothing(self):
        body = extract_function(self.code, "ouvrirLaCarte")
        for interdit in ("callAPI", "runOptimisation", "appendBenchmark",
                         "getPoints(", "setValue", "_saveCartePayload_",
                         "buildCartePayload"):
            self.assertNotIn(interdit, body,
                             "rouvrir la carte declenche %s" % interdit)


class TestShareableSnapshot(unittest.TestCase):
    """Un lien HTTPS pour consulter, un fichier HTML pour archiver.

    Drive ne PREVISUALISE pas le HTML qu'il n'a pas produit : il propose de
    le telecharger, ce qui ne mene nulle part d'utilisable sur telephone.
    L'instantane est donc servi par la Web App, qui est une vraie adresse
    HTTPS, et identifie par un jeton."""

    def setUp(self):
        self.code = read_code()
        self.html = read_map_html()

    # --- le jeton --------------------------------------------------------

    def test_the_token_is_long_and_random(self):
        body = extract_function(self.code, "_jetonPartage_")
        self.assertIn("Utilities.getUuid()", body)
        # Deux UUID : 64 caracteres hexadecimaux une fois les tirets retires.
        self.assertEqual(body.count("Utilities.getUuid()"), 2)
        self.assertIn('replace(/-/g, "")', body)
        for interdit in ("Math.random", "new Date().getTime()", "Date.now"):
            self.assertNotIn(interdit, body,
                             "le jeton est previsible : %s" % interdit)

    def test_the_registry_holds_the_snapshot_itself(self):
        """Le registre porte desormais l'instantane, pas un pointeur Drive.

        C'est ce qui supprime la lecture Drive du chemin de service."""
        entetes = header_list("SHARE_HEADERS")
        self.assertEqual(entetes, ["Jeton", "Créée le", "Expire le",
                                   "Nom", "Signature jeu", "Fragments"])
        self.assertNotIn("Fichier Drive", self.code)
        body = extract_function(self.code, "_enregistrerPartage_")
        self.assertNotIn("DriveApp", body)
        self.assertNotIn("fileId", body)

    def test_the_snapshot_is_split_under_the_cell_limit(self):
        """Une cellule plafonne a 50 000 caracteres ; deux traces routiers
        depassent largement."""
        self.assertIn("const SHARE_CHUNK = 45000;", self.code)
        body = extract_function(self.code, "_enregistrerPartage_")
        self.assertIn("texte.slice(i, i + SHARE_CHUNK)", body)
        self.assertIn("morceaux.length", body)
        # Un instantane aberrant est refuse net plutot qu'ecrit tronque.
        self.assertIn("SHARE_MAX_FRAGMENTS", body)
        self.assertIn("trop volumineux", body)
        lire = extract_function(self.code, "_lirePartage_")
        self.assertIn("SHARE_COL_FRAGMENTS, 1, nb", lire)

    def test_a_fragment_can_never_be_read_as_a_formula(self):
        """Un decoupage peut tomber sur « = », « + » ou « - » : Sheets y
        verrait une formule et corromprait la donnee en silence."""
        self.assertIn('const SHARE_FRAGMENT_PREFIX = "~";', self.code)
        ecrit = extract_function(self.code, "_enregistrerPartage_")
        self.assertIn("SHARE_FRAGMENT_PREFIX + texte.slice(", ecrit)
        self.assertIn('setNumberFormat("@")', ecrit)
        lit = extract_function(self.code, "_lirePartage_")
        self.assertIn("brut.slice(SHARE_FRAGMENT_PREFIX.length)", lit)

    def test_a_share_expires(self):
        self.assertIn("const SHARE_TTL_DAYS = 30;", self.code)
        enreg = extract_function(self.code, "_enregistrerPartage_")
        self.assertIn("SHARE_TTL_DAYS * 24 * 3600 * 1000", enreg)
        lire = extract_function(self.code, "_lirePartage_")
        self.assertIn("expire.getTime() < maintenant.getTime()", lire)
        self.assertIn("return null;", lire)

    def test_a_share_can_be_revoked(self):
        body = extract_function(self.code, "revoquerPartagesCarte")
        self.assertIn("deleteRows(2, n)", body)
        # La revocation ne detruit pas l'archive de l'utilisateur.
        for interdit in ("setTrashed", "removeFile", "DriveApp"):
            self.assertNotIn(interdit, body,
                             "la revocation supprime aussi le fichier : %s"
                             % interdit)
        # Aucune entree de menu ajoutee : le menu reste celui des tournees.
        flat = re.sub(r"\s+", " ", self.code)
        menu = flat[flat.index('ui.createMenu(MENU_RACINE_LABEL)'):]
        menu = menu[:menu.index(".addToUi()")]
        self.assertNotIn("revoquerPartagesCarte", menu)

    # --- le lien ---------------------------------------------------------

    def test_the_link_is_built_from_the_validated_web_app_url(self):
        export = extract_function(self.code, "exporterCarteDepuisDialogue")
        self.assertIn("_getWebAppUrl_()", export)
        self.assertIn('base + "?" + SHARE_PARAM + "="', export)
        self.assertIn("_enregistrerPartage_(enrichi", export)
        # Sans Web App deployee, pas de lien mort : un message.
        self.assertIn("info.shareError = MSG_WEB_APP_INDISPONIBLE;", export)

    def test_a_failed_link_does_not_lose_the_export(self):
        """Le fichier existe deja quand le lien est construit."""
        export = extract_function(self.code, "exporterCarteDepuisDialogue")
        self.assertLess(export.index("_deposerCarteSurDrive_("),
                        export.index("_getWebAppUrl_()"))
        self.assertIn("try {", export)
        self.assertIn("catch (e)", export)

    def test_the_drive_preview_url_is_never_offered_as_the_way_to_consult(self):
        partage = extract_function(self.html, "afficherPartage")
        self.assertIn("info.shareUrl", partage)
        self.assertIn("Télécharger l'archive", partage)
        # Le lien /view de Drive, celui qui affiche « Impossible d'ouvrir le
        # fichier », ne figure plus dans la fenetre.
        self.assertNotIn("info.url", partage)
        self.assertNotIn("Ouvrir dans Drive", strip_comments(self.html))

    # --- ce que voit l'ami -----------------------------------------------

    def test_the_shared_page_is_assembled_without_touching_drive(self):
        """Le seul maillon Drive du chemin de service a disparu."""
        body = extract_function(self.code, "doGet")
        self.assertIn("_buildStandaloneCarteHtml_(payload)", body)
        self.assertIn("HtmlService.createHtmlOutput(html)", body)
        for interdit in ("DriveApp", "drive.google.com", "/file/d/",
                         "uc?export=", "/view", "getDownloadUrl",
                         "setLocation", "window.location", "location.href",
                         "http-equiv", "refresh"):
            self.assertNotIn(interdit, strip_comments(body),
                             "doGet passe encore par %s" % interdit)
        # Servie responsive, comme le dialogue.
        self.assertIn("width=device-width, initial-scale=1, viewport-fit=cover",
                      body)

    def test_the_shared_page_needs_no_spreadsheet_and_no_backend(self):
        """Le gabarit autonome rend directement, sans google.script.run."""
        boot = self.html[self.html.index("function boot()"):]
        self.assertLess(boot.index("window.TOURNEES_PAYLOAD"),
                        boot.index("google.script.run"))
        self.assertIn("renderCarte(window.TOURNEES_PAYLOAD);", boot)
        # L'injection se fait a la construction, pas a la lecture.
        builder = extract_function(self.code, "_buildStandaloneCarteHtml_")
        self.assertIn("window.TOURNEES_PAYLOAD = ", builder)

    def test_the_shared_page_keeps_the_route_checkboxes_and_the_fit_button(self):
        """Elles vivent dans buildToggles, appele par renderCarte dans TOUS
        les contextes — la barre d'actions, elle, reste masquee."""
        render = extract_function(self.html, "renderCarte")
        self.assertIn("buildToggles();", render)
        self.assertIn("buildSummary(payload);", render)
        toggles = extract_function(self.html, "buildToggles")
        self.assertIn("Ajuster la vue", toggles)
        self.assertIn("type='checkbox' data-layer='", toggles)

    def test_the_snapshot_carries_the_geometry_already_loaded(self):
        self.assertIn("currentGeometries ? JSON.stringify(currentGeometries)",
                      self.html)
        export = extract_function(self.code, "exporterCarteDepuisDialogue")
        self.assertIn("_payloadAvecGeometries_(json, geometries)", export)
        # Aucun nouvel appel : ni optimisation, ni geometrie, ni ORS.
        for interdit in ("getCarteGeometrie", "UrlFetchApp", "callAPI",
                         "runOptimisation", "appendBenchmark", "getPoints("):
            self.assertNotIn(interdit, export,
                             "l'export declenche %s" % interdit)

    def test_no_secret_can_reach_the_shared_page(self):
        for content in (self.code, self.html):
            lowered = content.lower()
            for needle in ("ors_key", "ors_api_key", "api.openrouteservice.org",
                           "heigit.org", "authorization", "render.com/api",
                           "bearer "):
                self.assertNotIn(needle, lowered)

    # --- la fenetre de partage -------------------------------------------

    def test_the_share_panel_is_reduced_to_what_is_useful(self):
        """Le besoin est de COPIER un lien, pas de rouvrir la carte.

        « Ouvrir la carte partagee » ne faisait qu'afficher une seconde copie
        d'une carte deja sous les yeux — et c'est ce clic qui empruntait le
        parcours defectueux."""
        partage = extract_function(self.html, "afficherPartage")
        self.assertIn("Carte prête à être partagée", partage)
        self.assertIn("Copier le lien", partage)
        self.assertIn("Télécharger l'archive", partage)
        self.assertIn("Fermer", partage)
        self.assertIn("MSG_PARTAGE_AIDE", partage)
        self.assertNotIn("Ouvrir la carte partagée", self.html)
        # Le mot « map » ne remplace jamais « carte ».
        self.assertNotIn("map partag", self.html)

    def test_closing_the_share_panel_leaves_the_map_untouched(self):
        partage = extract_function(self.html, "afficherPartage")
        self.assertIn('getElementById("btn-close-share")', partage)
        self.assertIn('sortie.innerHTML = "";', partage)
        for interdit in ("renderCarte(", "map.remove", "location."):
            self.assertNotIn(interdit, partage)
        self.assertIn("Utilisez le lien pour consulter la carte sur ",
                      self.html)
        self.assertIn("ordinateur, iPhone, iPad ou Android.", self.html)

    def test_copying_always_leaves_a_way_out(self):
        """L'API presse-papiers est souvent refusee dans l'iframe : le champ
        reste affiche et selectionne, donc Ctrl+C fonctionne toujours."""
        body = extract_function(self.html, "copierLien")
        self.assertIn("champ.select();", body)
        self.assertIn('document.execCommand("copy")', body)
        self.assertIn("navigator.clipboard", body)
        self.assertIn("catch (e)", body)
        self.assertIn("Ctrl+C", body)

    def test_the_share_controls_are_touch_sized_and_contrasted(self):
        debut = self.html.index(".actions a.bouton{")
        regle = self.html[debut:self.html.index("}", debut)]
        for attendu in ("min-height:44px", "background:#ffffff",
                        "color:#0b57d0", "display:inline-flex",
                        "box-sizing:border-box", "max-width:100%"):
            self.assertIn(attendu, regle)
        debut = self.html.index(".partage input{")
        champ = self.html[debut:self.html.index("}", debut)]
        self.assertIn("min-height:44px", champ)
        self.assertIn("width:100%", champ)
        self.assertIn("box-sizing:border-box", champ)


class TestMapResponsive(unittest.TestCase):
    """Le panneau prenait 74 % d'un ecran de telephone en paysage, pour
    101 px de carte visible. Ces controles figent la correction."""

    def setUp(self):
        self.html = read_map_html()

    def test_the_map_comes_first_in_the_layout(self):
        layout = self.html[self.html.index('<div id="layout">'):]
        layout = layout[:layout.index("</div>\n</div>")]
        self.assertLess(layout.index('id="map"'), layout.index('id="panel"'))

    def test_dynamic_viewport_height_with_a_fallback(self):
        self.assertIn("height:100vh;height:100dvh", self.html)

    def test_iphone_safe_areas_are_honoured(self):
        self.assertIn("env(safe-area-inset-left)", self.html)
        self.assertIn("env(safe-area-inset-bottom)", self.html)

    def test_a_side_panel_on_desktop_a_bottom_drawer_on_phones(self):
        self.assertIn("flex:0 0 290px", self.html)
        self.assertIn("@media (max-width:820px), (max-height:520px)", self.html)
        self.assertIn("#panel.replie", self.html)

    def test_the_drawer_starts_collapsed_on_phones_only(self):
        body = extract_function(self.html, "initTiroir")
        self.assertIn('matchMedia("(max-width:820px), (max-height:520px)")', body)
        self.assertIn('classList.toggle("replie", petit)', body)

    def test_every_control_is_touch_sized(self):
        for regle in (".actions button{", ".toggles button{",
                      ".toggles label{", "details.tech>summary{"):
            bloc = self.html[self.html.index(regle):]
            bloc = bloc[:bloc.index("}")]
            self.assertIn("min-height:44px", bloc,
                          "%s n'est pas dimensionne pour le tactile" % regle)

    def test_the_map_is_resized_on_rotation(self):
        """Leaflet ne detecte pas seul un passage en paysage : sans
        invalidateSize la carte garde son ancienne taille."""
        self.assertIn('addEventListener("resize", recadrer)', self.html)
        self.assertIn('addEventListener("orientationchange"', self.html)
        body = extract_function(self.html, "recadrer")
        self.assertIn("map.invalidateSize()", body)
        self.assertIn("fitBounds", body)
        toggle = extract_function(self.html, "basculerTiroir")
        self.assertIn("recadrer", toggle)

    def test_technical_details_are_collapsed_by_default(self):
        self.assertIn('<details class="tech" id="technique">', self.html)
        self.assertNotIn('<details class="tech" id="technique" open>', self.html)
        bloc = self.html[self.html.index('id="technique"'):]
        bloc = bloc[:bloc.index("</details>")]
        for identifiant in ('id="legend"', 'id="meta"', 'id="territorial"'):
            self.assertIn(identifiant, bloc,
                          "%s devrait etre sous Details techniques" % identifiant)

    def test_nothing_is_removed_from_the_payload(self):
        """La strategie, la signature et les diagnostics territoriaux sont
        deplaces, jamais supprimes."""
        for champ in ("payload.strategy", "payload.points_signature",
                      "payload.generated_at", "buildTerritorial"):
            self.assertIn(champ, self.html)

    def test_the_two_dialog_actions_are_the_requested_ones(self):
        self.assertIn('id="btn-zoom"', self.html)
        self.assertIn(">Agrandir<", self.html)
        self.assertIn(">Exporter et partager<", self.html)
        # L'ancien bouton et son vocabulaire ont disparu : Apps Script ne
        # peut pas tenir la promesse d'un plein ecran de navigateur, et
        # « Ouvrir la carte » est le nom d'une entree de menu, pas d'un
        # bouton de la carte.
        # Sur le fichier SANS commentaires : celui qui explique le changement
        # doit forcement nommer ce qui a ete retire.
        rendu = strip_comments(self.html)
        for interdit in ("btn-fullscreen", "ouvrirEnGrand", "Plein écran",
                         "Ouvrir en grand", "requestFullscreen"):
            self.assertNotIn(interdit, rendu,
                             "%s subsiste dans la carte" % interdit)
        # « Ouvrir la carte » nomme une entree de MENU : elle ne doit pas
        # reapparaitre dans la barre d'actions du dialogue.
        barre = self.html[self.html.index('<div class="actions">'):]
        barre = barre[:barre.index("</div>")]
        self.assertNotIn("Ouvrir la carte", barre)

    def test_the_export_is_described_as_an_archive(self):
        self.assertIn("une archive HTML dans Google Drive", self.html)
        self.assertIn("Le fichier HTML sert ", self.html)
        self.assertIn("principalement d'archive.", self.html)


class TestEnlargeDialog(unittest.TestCase):
    """« Agrandir / Réduire » agit sur le CADRE du dialogue.

    L'ancien bouton ouvrait un onglet, ce qui ne repondait pas au besoin.
    requestFullscreen n'est pas fiable dans l'iframe sandboxee d'un dialogue
    Apps Script, et agrandir le contenu depuis l'interieur ne ferait que
    remplir un cadre inchange. google.script.host.setWidth / setHeight sont
    le seul agrandissement reel disponible ici."""

    def setUp(self):
        self.html = read_map_html()
        self.body = extract_function(self.html, "basculerAgrandissement")

    def test_the_button_is_a_toggle_with_both_labels(self):
        self.assertIn('id="btn-zoom"', self.html)
        self.assertIn(">Agrandir<", self.html)
        self.assertIn('bouton.textContent = agrandi ? "Réduire" : "Agrandir";',
                      self.body)
        self.assertIn('aria-pressed', self.html)
        self.assertIn('setAttribute("aria-pressed"', self.body)
        self.assertIn('getElementById("btn-zoom")\n'
                      '            .addEventListener("click", '
                      'basculerAgrandissement);', self.html)

    def test_it_resizes_the_dialog_and_never_opens_a_tab(self):
        self.assertIn("google.script.host.setWidth(largeur)", self.body)
        self.assertIn("google.script.host.setHeight(hauteur)", self.body)
        for interdit in ("window.open", "requestFullscreen", "_blank",
                         "getWebAppUrl", "location."):
            self.assertNotIn(interdit, self.body,
                             "l'agrandissement passe par %s" % interdit)

    def test_the_enlarged_size_asks_for_the_whole_available_screen(self):
        """Les anciens plafonds 1500x950 bridaient le dialogue bien en deca
        de ce que Sheets accorde sur un ecran large. On demande l'ecran, et
        Google ramene de lui-meme au maximum qu'il autorise."""
        self.assertIn("screen.availWidth", self.body)
        self.assertIn("screen.availHeight", self.body)
        self.assertIn("(screen.availWidth  || DIALOG_L) - 24", self.body)
        self.assertIn("(screen.availHeight || DIALOG_H) - 24", self.body)
        # Plus aucun plafond de notre cote. Sur le corps SANS commentaires :
        # celui qui explique le retrait doit citer les anciennes valeurs.
        sans = strip_comments(self.body)
        for plafond in ("1500", "950", "Math.min("):
            self.assertNotIn(plafond, sans,
                             "l'agrandissement est encore bride par %s"
                             % plafond)

    def test_the_panel_shrinks_so_the_map_takes_the_space(self):
        """Elargir le panneau irait contre le but de l'agrandissement."""
        debut = self.html.index("body.agrandi #panel{")
        regle = self.html[debut:self.html.index("}", debut)]
        self.assertIn("flex:0 0 250px", regle)
        # Le panneau normal reste plus large : c'est bien un retrecissement.
        debut = self.html.index("#panel{flex:0 0 ")
        normal = self.html[debut:self.html.index("}", debut)]
        self.assertIn("flex:0 0 290px", normal)

    def test_it_restores_the_original_dialog_size(self):
        self.assertIn("var DIALOG_L = 1200, DIALOG_H = 800;", self.html)
        self.assertIn("largeur = DIALOG_L;", self.body)
        self.assertIn("hauteur = DIALOG_H;", self.body)
        # Les memes dimensions que celles demandees par code.js.
        code = read_code()
        self.assertIn("const MAP_DIALOG_WIDTH  = 1200;", code)
        self.assertIn("const MAP_DIALOG_HEIGHT = 800;", code)

    def test_no_size_can_be_negative_or_unusably_small(self):
        self.assertIn("var DIALOG_MIN_L = 480, DIALOG_MIN_H = 380;", self.html)
        self.assertIn("largeur = Math.max(largeur, DIALOG_MIN_L);", self.body)
        self.assertIn("hauteur = Math.max(hauteur, DIALOG_MIN_H);", self.body)

    def test_a_refused_resize_does_not_break_the_display(self):
        self.assertIn("try {", self.body)
        self.assertIn("catch (e)", self.body)
        # Le libelle et la classe suivent quand meme : ils sont poses APRES
        # le bloc protege, donc un echec ne laisse pas l'etat incoherent.
        self.assertLess(self.body.index("catch (e)"),
                        self.body.index("agrandi = vise;"))

    def test_leaflet_is_told_and_the_view_follows_the_visible_routes(self):
        self.assertIn("map.invalidateSize(true)", self.body)
        self.assertIn("recadrer()", self.body)
        recadrer = extract_function(self.html, "recadrer")
        self.assertIn("latLngsVisibles()", recadrer)

    def test_the_enlarged_mode_adds_a_css_class(self):
        self.assertIn('document.body.classList.toggle("agrandi", agrandi);',
                      self.body)
        self.assertIn("body.agrandi #panel", self.html)


class TestActionButtonContrast(unittest.TestCase):
    """Le bouton principal etait ecrit en BLANC SUR BLANC.

    Deux regles se disputaient la barre d'actions :

      #actions button        -> specificite (1,0,1), background:#fff
      .actions button.primaire -> specificite (0,2,1), background:#1a73e8
                                                       color:#fff

    Le selecteur d'ID gagne sur `background`, qui redevenait blanc. Mais
    `color` n'avait aucun concurrent dans la regle d'ID : le blanc de
    .primaire s'appliquait. D'ou du blanc sur blanc, a taille correcte et
    parfaitement cliquable — donc invisible sans etre casse."""

    def setUp(self):
        self.html = read_map_html()

    def regle(self, selecteur):
        """Corps d'une regle CSS, du selecteur a son accolade fermante."""
        debut = self.html.index(selecteur + "{")
        return self.html[debut:self.html.index("}", debut)]

    def test_the_id_rule_that_caused_it_is_gone(self):
        """Plus aucun selecteur d'ID ne peint la barre d'actions."""
        self.assertNotIn("#actions button{", self.html)
        self.assertNotIn("#actions button:disabled{", self.html)

    def test_background_and_colour_are_always_declared_together(self):
        """Une regle qui pose l'un sans l'autre reouvre exactement la faille."""
        for selecteur in (".actions button", ".actions button:hover",
                          ".actions button.primaire",
                          ".actions button.primaire:hover"):
            corps = self.regle(selecteur)
            self.assertIn("background:", corps,
                          "%s ne declare pas de fond" % selecteur)
            self.assertIn("color:", corps,
                          "%s ne declare pas de couleur de texte" % selecteur)

    def test_the_two_states_are_contrasted(self):
        base = self.regle(".actions button")
        self.assertIn("background:#ffffff", base)
        self.assertIn("color:#0b57d0", base)
        primaire = self.regle(".actions button.primaire")
        self.assertIn("background:#0b57d0", primaire)
        self.assertIn("color:#ffffff", primaire)
        self.assertIn("border-color:#0b57d0", primaire)

    def test_the_button_is_touch_sized_and_never_clipped(self):
        base = self.regle(".actions button")
        for regle in ("min-height:44px", "padding:0 16px",
                      "display:inline-flex", "align-items:center",
                      "justify-content:center", "white-space:nowrap",
                      "overflow:visible", "font-weight:600",
                      "cursor:pointer", "box-sizing:border-box",
                      "max-width:100%"):
            self.assertIn(regle, base,
                          "le bouton d'action n'a pas %s" % regle)

    def test_hover_and_keyboard_focus_are_visible(self):
        self.assertIn(".actions button:hover{", self.html)
        self.assertIn(".actions button:focus{", self.html)
        self.assertIn(".actions button:focus-visible{", self.html)
        self.assertIn("outline:3px solid #0b57d0", self.html)

    def test_both_labels_fit_the_same_button(self):
        """« Agrandir » et « Réduire » partagent la meme boite : rien n'est
        dimensionne sur un libelle particulier."""
        base = self.regle(".actions button")
        self.assertIn("flex:1 1 auto", base)
        self.assertNotIn("width:", base.replace("max-width:", "")
                                       .replace("min-width:", ""))


class TestMapRouteToggles(unittest.TestCase):
    """Les deux cases suffisent a choisir les tournees visibles."""

    def setUp(self):
        self.html = read_map_html()

    def test_the_show_both_button_is_gone(self):
        # Sur le fichier SANS commentaires : le commentaire qui explique le
        # retrait cite forcement le libelle retire.
        rendu = strip_comments(self.html)
        for interdit in ("Afficher les deux", "showAll"):
            self.assertNotIn(interdit, rendu,
                             "le bouton %r est de retour" % interdit)

    def test_one_checkbox_per_route_still_drives_the_layers(self):
        body = extract_function(self.html, "buildToggles")
        self.assertIn("type='checkbox' data-layer='", body)
        self.assertIn("map.addLayer(g)", body)
        self.assertIn("map.removeLayer(g)", body)
        # Les libelles viennent des tournees, jamais d'une liste ecrite ici.
        self.assertIn("esc(layers[i].label)", body)

    def test_no_route_checked_is_said_not_crashed(self):
        body = extract_function(self.html, "majIndicationTournees")
        self.assertIn("MSG_AUCUNE_TOURNEE", body)
        self.assertIn('var MSG_AUCUNE_TOURNEE = "Sélectionnez au moins une '
                      'tournée.";', self.html)
        # Compte les cases, pas les couches : une tournee vide reste cochee.
        self.assertIn("boxes[i].checked", body)

    def test_the_fit_button_exists_and_recentres(self):
        body = extract_function(self.html, "buildToggles")
        self.assertIn("id='btn-fit'", body)
        self.assertIn("Ajuster la vue", body)
        self.assertIn('getElementById("btn-fit").addEventListener("click", '
                      "recadrer)", body)

    def test_recentring_only_considers_the_visible_routes(self):
        body = extract_function(self.html, "recadrer")
        self.assertIn("latLngsVisibles()", body)
        self.assertNotIn("allLatLngs", body)
        visible = extract_function(self.html, "latLngsVisibles")
        self.assertIn("map.hasLayer(layers[i].group)", visible)
        self.assertIn("layers[i].latlngs", visible)

    def test_the_edited_functions_are_balanced_and_defined_once(self):
        """Aucun interpreteur JavaScript n'est disponible dans ce depot.

        Le controle porte donc sur les fonctions REELLEMENT editees, ou il
        est exact : une accolade ou une parenthese manquante s'y verrait."""
        for name in ("drawRoute", "buildToggles", "majIndicationTournees",
                     "latLngsVisibles", "recadrer",
                     "basculerAgrandissement"):
            body = extract_function(self.html, name)
            self.assertTrue(body.strip(), "%s introuvable" % name)
            # extract_function s'arrete AVANT l'accolade fermante de la
            # fonction : on la remet, sinon tout corps parait desequilibre.
            sans = strip_comments(body) + "\n}"
            for ouvrant, fermant in (("(", ")"), ("[", "]"), ("{", "}")):
                self.assertEqual(
                    sans.count(ouvrant), sans.count(fermant),
                    "%s : %s et %s desequilibres" % (name, ouvrant, fermant))

        # Chaque fonction appelee par les cases a cocher existe, une fois.
        for helper in ("majIndicationTournees", "latLngsVisibles", "recadrer"):
            self.assertEqual(self.html.count("function %s(" % helper), 1,
                             "%s n'est pas defini une fois et une seule"
                             % helper)

    def test_every_route_carries_its_own_coordinates(self):
        """Sans coordonnees par tournee, le recentrage ne saurait pas quoi
        ignorer quand une case est decochee."""
        body = extract_function(self.html, "drawRoute")
        self.assertIn("var routeLatLngs = [];", body)
        self.assertIn("latlngs:routeLatLngs", body)
        # Y compris pour une tournee vide : la liste existe, elle est vide.
        self.assertIn("count:0, latlngs:[]", body)
        # allLatLngs reste alimente : le premier cadrage montre tout.
        self.assertIn("allLatLngs", body)

    def test_the_route_geometry_and_order_are_untouched(self):
        """Les cases changent la VISIBILITE, jamais les donnees."""
        for name in ("buildToggles", "majIndicationTournees",
                     "latLngsVisibles", "recadrer"):
            body = extract_function(self.html, name)
            for interdit in ("sort(", "reverse(", "splice(", ".lat =",
                             ".lon =", ".color =", "geometry ="):
                self.assertNotIn(interdit, body,
                                 "%s modifie les donnees de la tournee : %s"
                                 % (name, interdit))


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
            # Prefixe de schema, sans hote : sert a REFUSER une adresse
            # d'ouverture qui ne serait pas en HTTPS. Ne designe aucune
            # destination et ne peut pas en devenir une.
            "https://",
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
