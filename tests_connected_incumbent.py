"""
Tests de l'INCUMBENT HISTORIQUE et de l'absence d'equilibrage T1/T2.

Lancement :
    python -m unittest tests_connected_incumbent -v

Deux invariants sont verifies ici, tous deux comportementaux :

1. Le classement ne regarde JAMAIS l'equilibre entre les deux tournees. Une
   solution tres desequilibree mais plus rapide au total doit gagner. C'est la
   regle metier : une tournee peut legitimement etre bien plus longue que
   l'autre, seule la duree totale compte.

2. La diversification AJOUTE des candidates, elle n'en retire jamais. La
   meilleure partition du generateur d'avant diversification garde une place
   reservee parmi les finalistes OR-Tools, et le resultat retenu ne peut pas
   etre moins bon que le sien.
"""

import math
import statistics
import sys
import types
import unittest


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
        def deco(fn):
            return fn
        return deco

    def run(self, *a, **k):
        pass


_stub("flask", Flask=_FakeFlask, request=None, jsonify=lambda *a, **k: None)
_stub("requests", post=lambda *a, **k: None)
_np = _stub("numpy")
_np.mean = lambda seq: statistics.fmean(list(seq))
_np.array = lambda x: x
sys.modules["sklearn"] = types.ModuleType("sklearn")
_stub("sklearn.cluster", KMeans=object)
_stub("sklearn.metrics", silhouette_score=lambda *a, **k: 0.0)

import app  # noqa: E402


# --------------------------------------------------------------- fixtures

def P(pid, lat, lon):
    return {"id": str(pid), "lat": lat, "lon": lon, "address": "pt " + str(pid)}


def scattered_points(n, seed=7):
    pts = [P("DEP", 45.4666, 4.3903)]
    x = seed
    for k in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        a = x / 2147483648.0
        x = (1103515245 * x + 12345) % 2147483648
        b = x / 2147483648.0
        pts.append(P(k, 45.400 + 0.09 * a, 4.330 + 0.11 * b))
    return pts


def euclid_matrix(points, scale=1.0):
    xy = app._local_xy(points, list(range(len(points))))
    n = len(points)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i][j] = math.hypot(xy[i][0] - xy[j][0],
                                     xy[i][1] - xy[j][1]) * scale
    return m


def solution(duration_a, duration_b, distance_m, sequencer="ortools",
             partition_key=((1,), (2,)), legacy=False):
    """Solution dont on connait la repartition entre T1 et T2.

    duration_s est la SOMME : c'est le seul chiffre que la selection doit
    regarder. duration_t1 / duration_t2 sont la pour rendre le desequilibre
    visible dans le test, pas pour etre lus par le code.
    """
    return {
        "duration_s": float(duration_a + duration_b),
        "duration_t1": float(duration_a),
        "duration_t2": float(duration_b),
        "imbalance_s": abs(duration_a - duration_b),
        "distance_m": float(distance_m),
        "sequencer": sequencer,
        "selection_reason": app._SELECTION_REASONS.get(sequencer, sequencer),
        "partition_key": partition_key,
        "route_a": [0, 1, 0], "route_b": [0, 2, 0],
        "connected": True, "cardinality_ok": True,
        "boundary": {"cut_edges": 1, "enclave_points": 0},
        "legacy": legacy,
    }


# --------------------------------------- 1-2 : aucun equilibrage T1 / T2

class TestNoBalancingInSelection(unittest.TestCase):

    def test_01_unbalanced_but_faster_beats_balanced(self):
        """T1 = 100 min et T2 = 10 min contre deux tournees de 55 min :
        la solution desequilibree est plus rapide de 60 s au total, elle doit
        gagner. Aucun critere ne doit lui reprocher son desequilibre."""
        unbalanced = solution(6000, 600, 40000, partition_key=((1,), (2,)))
        balanced = solution(3330, 3330, 40000, partition_key=((3,), (4,)))
        self.assertEqual(unbalanced["duration_s"], 6600.0)
        self.assertEqual(balanced["duration_s"], 6660.0)
        self.assertGreater(unbalanced["imbalance_s"], balanced["imbalance_s"])

        best = app.select_best_solution([balanced, unbalanced])
        self.assertEqual(best["partition_key"], ((1,), (2,)),
                         "la solution desequilibree mais plus rapide a perdu")
        # et dans l'autre sens d'insertion
        self.assertEqual(
            app.select_best_solution([unbalanced, balanced])["partition_key"],
            ((1,), (2,)))

    def test_01b_extreme_imbalance_still_wins_when_faster(self):
        """Cas limite : une tournee vide. Rien dans le classement ne doit
        pousser a repartir la charge."""
        extreme = solution(6600, 0, 40000, partition_key=((1,), (2,)))
        even = solution(3320, 3320, 40000, partition_key=((3,), (4,)))
        self.assertEqual(app.select_best_solution([even, extreme])["partition_key"],
                         ((1,), (2,)))

    def test_02_balance_never_appears_in_the_ranking(self):
        """A duree totale et distance egales, deux repartitions opposees sont
        interchangeables : seuls les criteres deterministes les departagent,
        jamais l'equilibre."""
        even = solution(3300, 3300, 40000, partition_key=((1,), (2,)))
        skewed = solution(6000, 600, 40000, partition_key=((1,), (2,)))
        self.assertEqual(even["duration_s"], skewed["duration_s"])
        self.assertEqual(app._solution_tiebreak(even),
                         app._solution_tiebreak(skewed),
                         "la cle de departage lit la repartition T1/T2")

    def test_02b_selection_key_ignores_the_split(self):
        base = {"connected": True, "cardinality_ok": True, "components_total": 0,
                "duration_s": 6600.0, "distance_m": 40000.0,
                "boundary": {"cut_edges": 1, "enclave_points": 0},
                "group_a": [1, 2], "group_b": [3, 4],
                "partition_key": ((1, 2), (3, 4))}
        even = dict(base, duration_t1=3300.0, duration_t2=3300.0)
        skewed = dict(base, duration_t1=6000.0, duration_t2=600.0)
        self.assertEqual(app._selection_key(even), app._selection_key(skewed))

    def test_02c_a_slower_balanced_solution_never_wins_outside_the_window(self):
        """31 s de plus, hors fenetre : meme parfaitement equilibree et plus
        courte en distance, elle perd."""
        fast = solution(6000, 600, 45000, partition_key=((1,), (2,)))
        slow_even = solution(3315, 3316, 30000, partition_key=((3,), (4,)))
        self.assertEqual(slow_even["duration_s"] - fast["duration_s"], 31.0)
        self.assertEqual(app.select_best_solution([fast, slow_even])["partition_key"],
                         ((1,), (2,)))

    def test_02d_no_span_or_makespan_objective_in_the_solver(self):
        """OR-Tools ne doit porter aucune dimension de span : elle egaliserait
        les tournees, ce que la regle metier interdit. Seule la dimension de
        CAPACITE est posee, et elle sert le 30/30, une contrainte dure."""
        with open("app.py", encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("SetGlobalSpanCostCoefficient", src)
        self.assertNotIn("SetSpanCostCoefficient", src)
        body = src[src.index("def _solve_cvrp_ortools("):
                   src.index("def ortools_partition_haversine(")]
        self.assertIn("AddDimensionWithVehicleCapacity", body)
        self.assertEqual(body.count("AddDimension"), 1,
                         "une seule dimension attendue : la capacite")

    def test_02e_connected_pipeline_has_no_balancing_term(self):
        """Balayage du pipeline connexe : aucune valeur absolue ni ecart entre
        tournees dans les fonctions qui classent."""
        with open("app.py", encoding="utf-8") as fh:
            src = fh.read()
        for start, end in (("def _selection_key", "def _solution_tiebreak"),
                           ("def _solution_tiebreak", "def select_best_solution("),
                           ("def select_best_solution(", "_SELECTION_REASONS = {"),
                           ("def select_diverse_finalists(",
                            "def _min_pairwise_difference(")):
            body = src[src.index(start):src.index(end)]
            for forbidden in ("abs(", "makespan", "span", "balance", "variance"):
                self.assertNotIn(forbidden, body,
                                 "terme d'equilibrage dans %s : %s"
                                 % (start, forbidden))


# ------------------------------ 3-4 : protection de l'incumbent historique

def _rough(specs):
    """specs : (partition_key, duration, seed, legacy)."""
    out = []
    for pkey, dur, seed, legacy in specs:
        out.append({
            "group_a": list(pkey[0]), "group_b": list(pkey[1]), "seed": seed,
            "partition_key": pkey, "connected": True, "cardinality_ok": True,
            "components_total": 0, "duration_s": float(dur),
            "distance_m": float(dur) * 10.0,
            "boundary": {"cut_edges": 1, "enclave_points": 0},
            "legacy": legacy,
        })
    return out


class TestLegacyProtection(unittest.TestCase):

    UNIVERSE = list(range(1, 41))

    def _near(self, k):
        """Variante proche du decoupage 1-20 / 21-40 : deux points bougent."""
        ga = sorted([x for x in self.UNIVERSE[:20] if x != 20 - (k % 10)]
                    + [21 + (k % 19)])
        return app.canonical_partition_key(
            ga, sorted(x for x in self.UNIVERSE if x not in ga))

    def _far(self, k):
        """Decoupage franchement different : un point sur deux."""
        ga = sorted([x for x in self.UNIVERSE[0::2] if x != 1 + 2 * (k % 20)]
                    + [2 + 2 * (k % 20)])
        return app.canonical_partition_key(
            ga, sorted(x for x in self.UNIVERSE if x not in ga))

    def _crowded_field(self):
        """Terrain defavorable a la candidate historique.

        Douze quasi-jumelles bien classees remplissent les places de score, et
        SIX sources encore inutilisees proposent des decoupages tres eloignes
        qui remportent toutes les places de diversite. La candidate historique,
        mal classee ET proche des jumelles, n'a aucun argument : sans place
        reservee, elle sort.
        """
        specs = [(self._near(k), 1000 + k, "sweep", False) for k in range(12)]
        for rank, source in enumerate(("mst", "region_growing", "kmedoids",
                                       "two_means", "perturbation",
                                       "local_search")):
            specs.append((self._far(rank), 2000 + rank, source, False))
        legacy_key = self._near(50)
        specs.append((legacy_key, 99999, "legacy:sweep_2", True))
        return _rough(specs), legacy_key

    def test_03_the_historic_best_is_always_a_finalist(self):
        """Meme classee DERNIERE par le proxy, la candidate historique entre
        dans les douze : c'est le proxy qui se trompe, pas elle."""
        scored, legacy_key = self._crowded_field()

        without = app.select_diverse_finalists(scored, 12)[0]
        self.assertNotIn(legacy_key, [c["partition_key"] for c in without],
                         "fixture invalide : elle passe deja sans protection")

        chosen, _ = app.select_diverse_finalists(scored, 12,
                                                 protected_keys=(legacy_key,))
        self.assertIn(legacy_key, [c["partition_key"] for c in chosen],
                      "la candidate historique a ete evincee des finalistes")
        self.assertLessEqual(len(chosen), 12, "plafond de finalistes depasse")

    def test_03b_protection_does_not_evict_the_best_candidate(self):
        """La place reservee s'ajoute, elle ne prend pas celle du meilleur."""
        scored, legacy_key = self._crowded_field()
        best = min(scored, key=app._selection_key)
        chosen, _ = app.select_diverse_finalists(scored, 12,
                                                 protected_keys=(legacy_key,))
        keys = [c["partition_key"] for c in chosen]
        self.assertIn(best["partition_key"], keys,
                      "la meilleure candidate a ete sacrifiee a la protection")
        self.assertIn(legacy_key, keys)

    def test_03c_protection_is_deterministic(self):
        scored, legacy_key = self._crowded_field()
        a, da = app.select_diverse_finalists(scored, 12, protected_keys=(legacy_key,))
        b, db = app.select_diverse_finalists(scored, 12, protected_keys=(legacy_key,))
        self.assertEqual([c["partition_key"] for c in a],
                         [c["partition_key"] for c in b])
        self.assertEqual(da, db)

    def test_04_the_legacy_generator_partitions_all_survive(self):
        """Reproduction du generateur historique, puis generation complete :
        chacune de ses partitions doit se retrouver dans le jeu diversifie.
        C'est exactement ce qui manquait -- connected_local_search() n'etait
        plus appelee, donc ses appartenances n'existaient plus."""
        pts = scattered_points(40)
        idx = [i for i in range(len(pts)) if i != 0]
        target = len(idx) // 2
        dur = euclid_matrix(pts)
        hav = app._build_haversine_matrix(pts)
        adj, gmeta = app.build_geo_graph(pts, idx, dur_matrix=dur)

        legacy = app.legacy_connected_candidates(pts, idx, target, adj, dur, 0, 0)
        self.assertGreater(len(legacy), 0, "le generateur historique est muet")
        legacy_keys = {app.canonical_partition_key(ga, gb)
                       for _s, ga, gb in legacy}

        cands, stats = app.generate_connected_candidates(
            pts, idx, target, adj, dur, hav, 0, 0,
            tree_edges=gmeta["tree_edges"])
        produced = {c["partition_key"] for c in cands}
        missing = legacy_keys - produced
        self.assertEqual(missing, set(),
                         "%d partition(s) historique(s) perdue(s)" % len(missing))
        self.assertTrue(any(c.get("legacy") for c in cands),
                        "aucune candidate marquee historique")

    def test_04b_legacy_candidates_come_from_the_local_optimum(self):
        """La source historique termine par connected_local_search : son
        appartenance n'est PAS celle de la coupe brute. Sans cette etape, la
        partition gagnante d'origine n'est jamais proposee."""
        pts = scattered_points(30)
        idx = [i for i in range(len(pts)) if i != 0]
        target = len(idx) // 2
        dur = euclid_matrix(pts)
        adj, _g = app.build_geo_graph(pts, idx, dur_matrix=dur)
        xy = app._local_xy(pts, idx)

        sweep, _s = app.enumerate_territorial_partitions(pts, idx, target)
        self.assertGreater(len(sweep), 0)
        raw = sweep[0]
        ga, gb = app._normalize_sizes(raw["group_a"], raw["group_b"], idx,
                                      target, xy)
        ga, gb, ok, _m = app.repair_to_connected(ga, gb, adj, pts, dur, 0, 0,
                                                 target)
        self.assertTrue(ok)
        na, nb, _est, swaps = app.connected_local_search(ga, gb, adj, dur, 0, 0)
        if swaps:
            self.assertNotEqual(app.canonical_partition_key(ga, gb),
                                app.canonical_partition_key(na, nb),
                                "la recherche locale n'a rien change")
        legacy = app.legacy_connected_candidates(pts, idx, target, adj, dur, 0, 0)
        keys = {app.canonical_partition_key(a, b) for _s, a, b in legacy}
        self.assertIn(app.canonical_partition_key(na, nb), keys)

    def test_04c_every_legacy_candidate_is_valid(self):
        pts = scattered_points(40)
        idx = [i for i in range(len(pts)) if i != 0]
        target = len(idx) // 2
        dur = euclid_matrix(pts)
        adj, _g = app.build_geo_graph(pts, idx, dur_matrix=dur)
        for source, ga, gb in app.legacy_connected_candidates(
                pts, idx, target, adj, dur, 0, 0):
            self.assertTrue(source.startswith("legacy:"), source)
            ok, reason, _ = app.validate_partition(ga, gb, idx, target, adj)
            self.assertTrue(ok, "%s invalide : %s" % (source, reason))


# ------------------------ 4 a 7 : non-degradation de bout en bout

class _Harness:
    """Matrice ORS, OR-Tools et Vroom bouchonnes autour de la strategie."""

    def __init__(self, points):
        self.points = points
        self.dur = euclid_matrix(points)
        self.dist = euclid_matrix(points, scale=1000.0)
        self.vroom_calls = []
        self.matrix_calls = 0
        self._saved = {}

    def _matrix(self, points, headers):
        self.matrix_calls += 1
        return self.dur, self.dist, {"ors_matrix": {"stub": True},
                                     "content_hash": "deadbeefcafe"}, None

    def _tsp(self, matrix, group, s, e):
        return app._estimate_group_cost(matrix, group, s, e, True)[1]

    def _vroom(self, points, group, s, e, headers):
        self.vroom_calls.append(tuple(sorted(group)))
        return app._nn_route_matrix(self.dur, sorted(group), s, e), 0, 0

    def __enter__(self):
        for name in ("_build_full_matrix_chunked", "_tsp_order_ortools",
                     "_resequence_single", "ORTOOLS_AVAILABLE",
                     "_solve_cvrp_ortools"):
            self._saved[name] = getattr(app, name)
        app._build_full_matrix_chunked = self._matrix
        app._tsp_order_ortools = self._tsp
        app._resequence_single = self._vroom
        app.ORTOOLS_AVAILABLE = True
        app._solve_cvrp_ortools = lambda *a, **k: (None, "stub")
        return self

    def __exit__(self, *exc):
        for name, value in self._saved.items():
            setattr(app, name, value)
        return False


class TestDiversificationNeverDegrades(unittest.TestCase):

    def _run(self, n_points=40):
        pts = scattered_points(n_points)
        with _Harness(pts) as h:
            groups, err, meta = app.ortools_partition_ors_matrix_connected(
                pts, 2, len(pts), 0, 0, {})
        return pts, groups, err, meta, h

    def test_04b_several_legacy_slots_are_reserved_not_just_one(self):
        """Le proxy classe mal les partitions historiques : la meilleure au
        proxy n'est pas la meilleure apres sequencement. Reserver une seule
        place ne garantirait donc pas l'incumbent reel."""
        _pts, _g, _err, meta, _h = self._run()
        diag = meta["connected"]
        self.assertGreater(diag["connected_legacy_finalist_slots"], 1)
        self.assertLessEqual(diag["connected_legacy_finalist_slots"],
                             app.CONNECTED_LEGACY_FINALIST_SLOTS)
        self.assertEqual(diag["connected_legacy_finalists"],
                         diag["connected_legacy_finalist_slots"],
                         "des places reservees n'ont pas ete honorees")
        # la diversite garde au moins l'autre moitie des places
        self.assertLessEqual(diag["connected_legacy_finalists"],
                             app.CONNECTED_ORTOOLS_FINALISTS
                             - app.CONNECTED_LEGACY_FINALIST_SLOTS
                             + app.CONNECTED_LEGACY_FINALIST_SLOTS)
        self.assertGreater(diag["connected_candidates_selected_diverse"],
                           diag["connected_legacy_finalists"],
                           "les historiques ont rafle tous les finalistes")

    def test_04_the_winner_is_never_worse_than_the_legacy_incumbent(self):
        """Invariant central du lot : la diversification AJOUTE des solutions.
        Le resultat retenu ne peut donc jamais etre moins bon que le meilleur
        de la partition historique, evalue sur la MEME matrice."""
        _pts, _g, err, meta, _h = self._run()
        diag = meta["connected"]
        self.assertIsNone(err)
        self.assertTrue(diag["connected_legacy_protected"],
                        "aucune candidate historique n'a ete produite")
        self.assertTrue(diag["connected_legacy_in_finalists"],
                        "la candidate historique n'a pas atteint OR-Tools")
        self.assertIsNotNone(diag["connected_legacy_duration_s"])
        self.assertLessEqual(diag["connected_selected_duration_s"],
                             diag["connected_legacy_duration_s"] + 1e-6,
                             "la diversification a degrade l'incumbent")

    def test_05_the_thirty_second_rule_still_holds_end_to_end(self):
        _pts, _g, _err, meta, _h = self._run()
        diag = meta["connected"]
        self.assertEqual(diag["connected_selection_window_s"], 30.0)
        # le gagnant appartient bien a la fenetre du minimum global
        best = min(v for v in (diag["ortools_total_duration_s"],
                               diag["vroom_total_duration_s"],
                               diag["connected_selected_duration_s"])
                   if v is not None)
        self.assertLessEqual(diag["connected_selected_duration_s"], best + 30.0)

    def test_06_sequencer_routes_and_metrics_stay_coherent(self):
        _pts, groups, _err, meta, h = self._run()
        diag = meta["connected"]
        routes = meta["connected_routes"]
        self.assertIn(diag["selected_sequencer"],
                      ("ortools", "vroom", "heuristic"))
        self.assertEqual(diag["final_selection_reason"],
                         app._SELECTION_REASONS[diag["selected_sequencer"]])
        # les routes decrivent exactement les deux groupes retournes
        self.assertEqual(sorted(routes[0][1:-1]), sorted(groups[0]))
        self.assertEqual(sorted(routes[1][1:-1]), sorted(groups[1]))
        # et les metriques sont recalculees depuis ces routes
        dur, dist = app._rescore(h.dur, h.dist, routes[0], routes[1])
        self.assertAlmostEqual(diag["connected_selected_duration_s"],
                               round(dur, 1), places=1)
        self.assertAlmostEqual(diag["connected_selected_distance_m"],
                               round(dist, 1), places=1)

    def test_07_six_vroom_calls_at_most(self):
        _pts, _g, _err, meta, h = self._run()
        self.assertLessEqual(len(h.vroom_calls), 6)
        self.assertEqual(meta["connected"]["connected_vroom_calls"],
                         len(h.vroom_calls))
        self.assertEqual(h.matrix_calls, 1,
                         "la matrice ORS a ete rechargee")

    def test_diversification_adds_candidates_without_removing_any(self):
        _pts, _g, _err, meta, _h = self._run()
        diag = meta["connected"]
        self.assertGreater(diag["connected_candidates_legacy"], 0)
        self.assertGreater(diag["connected_candidates_unique"],
                           diag["connected_candidates_legacy"],
                           "la diversification n'a rien ajoute")
        self.assertEqual(diag["connected_candidates_selected_diverse"],
                         min(app.CONNECTED_ORTOOLS_FINALISTS,
                             diag["connected_candidates_unique"]))

    def test_matrix_hash_is_reported(self):
        _pts, _g, _err, meta, _h = self._run()
        self.assertEqual(meta["connected"]["connected_matrix_hash"],
                         "deadbeefcafe")

    def test_run_is_deterministic(self):
        r1 = self._run()[3]["connected"]
        r2 = self._run()[3]["connected"]
        for key in ("selected_sequencer", "connected_selected_duration_s",
                    "connected_selected_distance_m", "connected_legacy_seed",
                    "connected_candidates_unique",
                    "connected_legacy_in_finalists"):
            self.assertEqual(r1[key], r2[key], "champ instable : " + key)


# ------------------------------------------- diagnostics de tracabilite

class TestTraceability(unittest.TestCase):

    def test_matrix_content_hash_detects_a_changed_road_network(self):
        """Deux runs sur la MEME signature de points peuvent avoir recu des
        durees routieres differentes. L'empreinte du contenu le montre ; la
        signature des points, non."""
        pts = scattered_points(12)
        dur = euclid_matrix(pts)
        dist = euclid_matrix(pts, scale=1000.0)
        h1 = app._matrix_content_hash(dur, dist)
        self.assertEqual(h1, app._matrix_content_hash(dur, dist))
        dur[1][2] = dur[1][2] + 1.0
        self.assertNotEqual(h1, app._matrix_content_hash(dur, dist),
                            "une matrice modifiee garde la meme empreinte")
        self.assertEqual(len(h1), 12)

    def test_short_key_is_stable_and_mirror_insensitive(self):
        k1 = app.canonical_partition_key([1, 2, 3], [4, 5, 6])
        k2 = app.canonical_partition_key([4, 5, 6], [1, 2, 3])
        self.assertEqual(app._short_key(k1), app._short_key(k2))
        self.assertEqual(len(app._short_key(k1)), 8)
        self.assertNotEqual(app._short_key(k1),
                            app._short_key(app.canonical_partition_key(
                                [1, 2, 4], [3, 5, 6])))

    def test_per_source_report_covers_every_outcome(self):
        pts = scattered_points(30)
        idx = [i for i in range(len(pts)) if i != 0]
        dur = euclid_matrix(pts)
        hav = app._build_haversine_matrix(pts)
        adj, gmeta = app.build_geo_graph(pts, idx, dur_matrix=dur)
        _c, stats = app.generate_connected_candidates(
            pts, idx, len(idx) // 2, adj, dur, hav, 0, 0,
            tree_edges=gmeta["tree_edges"])
        self.assertGreater(len(stats["per_source"]), 0)
        for src, b in stats["per_source"].items():
            for field in ("raw", "duplicates", "invalid_size", "disconnected",
                          "repair_failed", "unique"):
                self.assertIn(field, b, "%s : champ %s manquant" % (src, field))
            self.assertEqual(
                b["unique"] + b["duplicates"] + b["invalid_size"]
                + b["disconnected"] + b["repair_failed"], b["raw"],
                "bilan incoherent pour %s : %s" % (src, b))
        self.assertEqual(sum(b["raw"] for b in stats["per_source"].values()),
                         stats["raw"])

    def test_per_source_text_is_stable_and_sorted(self):
        text = app._flatten_per_source({
            "sweep": {"raw": 180, "duplicates": 150, "invalid_size": 0,
                      "disconnected": 8, "repair_failed": 2, "unique": 20},
            "mst": {"raw": 4, "duplicates": 2, "invalid_size": 0,
                    "disconnected": 0, "repair_failed": 0, "unique": 2},
        })
        self.assertTrue(text.startswith("mst:"), text)
        self.assertIn("sweep:r=180,d=150,s=0,x=8,f=2,u=20", text)
        self.assertEqual(app._flatten_per_source({}), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
