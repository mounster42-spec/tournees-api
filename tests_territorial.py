"""
Tests de la partition territoriale de la strategie ortools_ors_matrix.

Lancement :
    python -m unittest tests_territorial -v

Aucun acces reseau, aucune dependance de production installee : flask, numpy,
requests et sklearn sont bouchonnes avant l'import d'app.py. ortools est
volontairement laisse absent, app.py basculant alors ORTOOLS_AVAILABLE a False
sans consequence pour la geometrie testee ici.
"""

import math
import re
import statistics
import sys
import types
import unittest


# ----------------------------------------------------------------- bouchons

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


# ----------------------------------------------------------------- fixtures

def P(pid, lat, lon):
    return {"id": str(pid), "lat": lat, "lon": lon, "address": "pt " + str(pid)}


def two_clusters(per_side=3, gap=0.02):
    """Depot en 0, puis deux amas nettement separes en longitude."""
    pts = [P("DEP", 45.50, 4.40)]
    for k in range(per_side):
        pts.append(P("W%d" % k, 45.44 + 0.001 * k, 4.38))
    for k in range(per_side):
        pts.append(P("E%d" % k, 45.44 + 0.001 * k, 4.38 + gap))
    return pts


def deliveries(points, start_idx=0, end_idx=0):
    depots = {start_idx, end_idx}
    return [i for i in range(len(points)) if i not in depots]


def flat_matrix(n, value=100.0):
    """Matrice carree constante, diagonale nulle."""
    m = [[value] * n for _ in range(n)]
    for i in range(n):
        m[i][i] = 0.0
    return m


def euclid_matrix(points):
    """Matrice de couts proportionnelle a la distance euclidienne locale."""
    xy = app._local_xy(points, list(range(len(points))))
    n = len(points)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = xy[i][0] - xy[j][0]
                dy = xy[i][1] - xy[j][1]
                m[i][j] = math.hypot(dx, dy)
    return m


# ------------------------------------------------------------------- tests

class TestCertificate(unittest.TestCase):

    def test_01_two_natural_groups_are_separated(self):
        pts = two_clusters(3)
        idx = deliveries(pts)
        cands, stats = app.enumerate_territorial_partitions(pts, idx, 3)
        self.assertGreater(len(cands), 0, "aucune separation trouvee")
        for c in cands:
            self.assertEqual(c["violations"], 0)
            self.assertGreater(c["margin_m"], 0.0)

        # la coupure ouest/est doit figurer parmi les candidates
        west = tuple(sorted(i for i in idx if pts[i]["id"].startswith("W")))
        keys = {app._partition_key(c["group_a"], c["group_b"]) for c in cands}
        self.assertIn(west, keys, "la separation naturelle ouest/est est absente")

    def test_02_north_south_axis(self):
        pts = [P("DEP", 45.50, 4.40)]
        for k in range(3):
            pts.append(P("S%d" % k, 45.40 + 0.001 * k, 4.40 + 0.001 * k))
        for k in range(3):
            pts.append(P("N%d" % k, 45.46 + 0.001 * k, 4.40 + 0.001 * k))
        idx = deliveries(pts)
        cands, _ = app.enumerate_territorial_partitions(pts, idx, 3)
        south = tuple(sorted(i for i in idx if pts[i]["id"].startswith("S")))
        keys = {app._partition_key(c["group_a"], c["group_b"]) for c in cands}
        self.assertIn(south, keys)

    def test_03_east_west_axis(self):
        pts = two_clusters(4, gap=0.05)
        idx = deliveries(pts)
        cands, _ = app.enumerate_territorial_partitions(pts, idx, 4)
        west = tuple(sorted(i for i in idx if pts[i]["id"].startswith("W")))
        keys = {app._partition_key(c["group_a"], c["group_b"]) for c in cands}
        self.assertIn(west, keys)

    def test_04_collinear_points(self):
        # 6 points parfaitement alignes : une seule coupure 3/3 est possible.
        pts = [P("DEP", 45.50, 4.40)]
        for k in range(6):
            pts.append(P("C%d" % k, 45.40 + 0.002 * k, 4.40 + 0.002 * k))
        idx = deliveries(pts)
        cands, _ = app.enumerate_territorial_partitions(pts, idx, 3)
        self.assertGreater(len(cands), 0, "des points colineaires restent separables")
        self.assertEqual(len(cands), 1, "une seule coupure 3/3 est geometriquement possible")
        for c in cands:
            self.assertEqual(c["violations"], 0)

    def test_05_identical_projections_are_rejected(self):
        # Deux points confondus a la frontiere : la marge est nulle, aucune
        # droite ne les separe, la coupure doit etre refusee.
        xy = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (100.0, 0.0), 3: (100.0, 0.0)}
        ga, gb, margin = app._split_by_angle(xy, [0, 1, 2, 3], 0.0, 1)
        self.assertIsNone(ga, "une marge nulle doit etre refusee")
        self.assertEqual(margin, 0.0)

    def test_06_duplicated_coordinates(self):
        pts = [P("DEP", 45.50, 4.40),
               P("A", 45.44, 4.38), P("B", 45.44, 4.38),   # confondus
               P("C", 45.44, 4.42), P("D", 45.44, 4.42)]   # confondus
        idx = deliveries(pts)
        cands, _ = app.enumerate_territorial_partitions(pts, idx, 2)
        self.assertGreater(len(cands), 0, "deux paires confondues restent separables")
        for c in cands:
            self.assertEqual(c["violations"], 0)
            self.assertGreater(c["margin_m"], 0.0)

    def test_07_determinism(self):
        pts = two_clusters(8)
        idx = deliveries(pts)
        a, sa = app.enumerate_territorial_partitions(pts, idx, 8)
        b, sb = app.enumerate_territorial_partitions(pts, idx, 8)
        self.assertEqual(sa, sb)
        self.assertEqual([app._partition_key(c["group_a"], c["group_b"]) for c in a],
                         [app._partition_key(c["group_a"], c["group_b"]) for c in b])

        mat = euclid_matrix(pts)
        s1, _ = app.select_territorial_partition(a, mat, mat, 0, 0)
        s2, _ = app.select_territorial_partition(b, mat, mat, 0, 0)
        self.assertEqual(sorted(s1["group_a"]), sorted(s2["group_a"]))
        self.assertEqual(s1["est_duration_s"], s2["est_duration_s"])

    def test_13_partitions_are_deduplicated(self):
        pts = two_clusters(5)
        idx = deliveries(pts)
        cands, stats = app.enumerate_territorial_partitions(pts, idx, 5)
        keys = [app._partition_key(c["group_a"], c["group_b"]) for c in cands]
        self.assertEqual(len(keys), len(set(keys)), "doublons dans les candidates")
        self.assertEqual(stats["unique"], len(cands))
        self.assertLess(stats["unique"], stats["generated"],
                        "la deduplication doit reduire le nombre de coupures")

    def test_certificate_counts_violations_on_overlap(self):
        # Groupes volontairement imbriques : le certificat doit les rejeter.
        xy = {0: (0.0, 0.0), 1: (50.0, 0.0), 2: (25.0, 0.0), 3: (75.0, 0.0)}
        viol, margin, _ = app._territorial_certificate(xy, [0, 1], [2, 3], 0.0)
        self.assertGreater(viol, 0, "un chevauchement doit produire des violations")
        self.assertLessEqual(margin, 0.0)


class TestSixtyPoints(unittest.TestCase):
    """Cas de production : 60 points de collecte, depart et arrivee communs."""

    def setUp(self):
        self.pts = [P("DEP", 45.4666, 4.3903)]
        for k in range(30):
            self.pts.append(P("T1_%d" % k, 45.4400 - 0.0012 * k, 4.3800 + 0.0009 * k))
        for k in range(30):
            self.pts.append(P("T2_%d" % k, 45.4200 - 0.0012 * k, 4.3920 + 0.0009 * k))
        self.idx = deliveries(self.pts)
        self.mat = euclid_matrix(self.pts)

    def test_08_exactly_30_30(self):
        cands, stats = app.enumerate_territorial_partitions(self.pts, self.idx, 30)
        self.assertGreater(len(cands), 0)
        for c in cands:
            self.assertEqual(len(c["group_a"]), 30)
            self.assertEqual(len(c["group_b"]), 30)

    def test_09_10_11_no_loss_no_duplicate_zero_violations(self):
        cands, _ = app.enumerate_territorial_partitions(self.pts, self.idx, 30)
        allset = set(self.idx)
        for c in cands:
            a, b = set(c["group_a"]), set(c["group_b"])
            self.assertEqual(a | b, allset, "point perdu")
            self.assertEqual(a & b, set(), "point duplique")
            self.assertEqual(c["violations"], 0, "violation territoriale")
            self.assertGreater(c["margin_m"], 0.0)

    def test_12_depots_excluded(self):
        cands, _ = app.enumerate_territorial_partitions(self.pts, self.idx, 30)
        for c in cands:
            self.assertNotIn(0, c["group_a"])
            self.assertNotIn(0, c["group_b"])

    def test_selection_respects_lexicographic_order(self):
        cands, _ = app.enumerate_territorial_partitions(self.pts, self.idx, 30)
        chosen, stats = app.select_territorial_partition(cands, self.mat, self.mat, 0, 0)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["violations"], 0)
        self.assertEqual(stats["scored"], len(cands))
        self.assertLessEqual(stats["refined"], app.TERRITORIAL_TOP_REFINE)

        # Le choix doit etre le meilleur des affinees, pas un hasard.
        best = None
        for c in cands:
            da, _ = app._estimate_group_cost(self.mat, c["group_a"], 0, 0, False)
            db, _ = app._estimate_group_cost(self.mat, c["group_b"], 0, 0, False)
            if best is None or da + db < best:
                best = da + db
        self.assertLessEqual(chosen["est_duration_s"], best * 1.0000001,
                             "l'affinage ne doit jamais degrader l'estimation brute")

    def test_no_balancing_term_in_objective(self):
        """Le cout ne doit contenir aucun terme d'equilibrage : une partition
        tres desequilibree en duree doit pouvoir gagner si son total est bon."""
        with open("app.py", encoding="utf-8") as fh:
            src = fh.read()
        body = src[src.index("def select_territorial_partition"):
                   src.index("def ortools_partition_ors_matrix")]
        self.assertNotIn("abs(", body,
                         "aucune valeur absolue d'ecart ne doit entrer dans le cout")


class TestLockAndStructure(unittest.TestCase):

    def setUp(self):
        with open("app.py", encoding="utf-8") as fh:
            self.src = fh.read()

    def test_14_membership_lock_flag_exists(self):
        self.assertIn("territorial_membership_locked", self.src)
        self.assertIn("membership_locked = True", self.src)
        self.assertIn("territorial_partition_locked", app.SWAP_STOP_REASONS)

    def test_15_swaps_not_called_when_locked(self):
        """Verification STRUCTURELLE : la branche verrouillee precede la branche
        des swaps et ne les appelle pas. L'appel effectif de /optimize demande
        un contexte Flask complet, hors perimetre de ces tests hors reseau."""
        locked_at = self.src.index("if routes_idx and vroom_ok and membership_locked:")
        swaps_at = self.src.index("elif routes_idx and vroom_ok:")
        self.assertLess(locked_at, swaps_at,
                        "la branche verrouillee doit etre testee en premier")
        branch = self.src[locked_at:swaps_at]
        self.assertNotIn("post_process_swaps(", branch,
                         "aucun appel de swaps dans la branche verrouillee")
        self.assertIn('"territorial_partition_locked"', branch)

    def test_16_kmeans_untouched(self):
        for name in ("kmeans_partition", "_balance_groups",
                     "_create_sub_clusters", "_enumerate_partitions"):
            self.assertIn("def " + name + "(", self.src)

    def test_17_haversine_untouched(self):
        self.assertIn("def ortools_partition_haversine(", self.src)
        self.assertIn("def _solve_cvrp_ortools(", self.src)
        body = self.src[self.src.index("def ortools_partition_haversine"):
                        self.src.index("def _matrix_cache_key")]
        self.assertIn("_solve_cvrp_ortools(", body,
                      "haversine doit continuer a passer par le solveur OR-Tools")
        self.assertNotIn("territorial", body,
                         "haversine ne doit pas dependre de la logique territoriale")

    def test_invariants_preserved(self):
        self.assertIn("ORTOOLS_SOLUTION_LIMIT = 75", self.src)
        self.assertIn("ORTOOLS_TIME_LIMIT_S = 25", self.src)
        self.assertIn("api.heigit.org/vroom/v0", self.src)
        self.assertIn("api.heigit.org/openrouteservice/v2/matrix/driving-car", self.src)
        self.assertIn("key = tuple(pts)", self.src)          # cache D-3 exact
        self.assertIn("SWAP_MAX_CANDIDATES = 50", self.src)
        self.assertIn("duration_s", self.src)                # garde-fou D-2


class TestRobustness(unittest.TestCase):

    def test_odd_and_small_counts(self):
        # 5 points : 2/3, la repartition la plus egale possible.
        pts = [P("DEP", 45.50, 4.40)]
        for k in range(5):
            pts.append(P("X%d" % k, 45.40 + 0.003 * k, 4.40))
        idx = deliveries(pts)
        cands, _ = app.enumerate_territorial_partitions(pts, idx, 2)
        self.assertGreater(len(cands), 0)
        for c in cands:
            self.assertEqual(len(c["group_a"]), 2)
            self.assertEqual(len(c["group_b"]), 3)

    def test_too_few_points(self):
        cands, stats = app.enumerate_territorial_partitions(
            [P("DEP", 45.5, 4.4), P("A", 45.4, 4.4)], [1], 1)
        self.assertEqual(cands, [])

    def test_invalid_coordinates_detected(self):
        self.assertFalse(app._finite_coords({"lat": None, "lon": 4.4}))
        self.assertFalse(app._finite_coords({"lat": "abc", "lon": 4.4}))
        self.assertFalse(app._finite_coords({"lat": 200.0, "lon": 4.4}))
        self.assertFalse(app._finite_coords({"lat": float("nan"), "lon": 4.4}))
        self.assertFalse(app._finite_coords({"lat": float("inf"), "lon": 4.4}))
        self.assertTrue(app._finite_coords({"lat": 45.4, "lon": 4.4}))

    def test_all_points_identical_has_no_separator(self):
        pts = [P("DEP", 45.50, 4.40)] + [P("S%d" % k, 45.44, 4.38) for k in range(4)]
        idx = deliveries(pts)
        cands, stats = app.enumerate_territorial_partitions(pts, idx, 2)
        self.assertEqual(cands, [], "des points tous confondus ne sont pas separables")
        self.assertEqual(stats["unique"], 0)

    def test_nn_route_is_order_independent(self):
        """Les egalites de cout sont departagees par index : le resultat ne
        depend pas de l'ordre d'iteration de la collection d'entree."""
        m = flat_matrix(5, 10.0)
        r1 = app._nn_route_matrix(m, [1, 2, 3, 4], 0, 0)
        r2 = app._nn_route_matrix(m, [4, 3, 2, 1], 0, 0)
        self.assertEqual(r1, r2)

    def test_distinct_start_and_end(self):
        pts = two_clusters(3)
        pts.append(P("ARR", 45.55, 4.45))
        end_idx = len(pts) - 1
        idx = deliveries(pts, 0, end_idx)
        self.assertNotIn(0, idx)
        self.assertNotIn(end_idx, idx)
        cands, _ = app.enumerate_territorial_partitions(pts, idx, 3)
        self.assertGreater(len(cands), 0)
        for c in cands:
            self.assertNotIn(end_idx, c["group_a"])
            self.assertNotIn(end_idx, c["group_b"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
