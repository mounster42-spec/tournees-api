"""Tests de l'endpoint isole /map-geometry.

Cet endpoint ne sert qu'a TRACER la carte. Ce qui est verifie ici tient en
deux idees :

  - il ne doit jamais devenir un proxy ORS generique : tout ce qui n'est pas
    explicitement autorise est refuse ;
  - il ne doit avoir AUCUN effet sur l'optimisation : ni compteur partage,
    ni cache partage, ni appel depuis /optimize.

Aucun reseau : requests.post est remplace par un double deterministe.
"""

import json
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


# =========================================================================
# DOUBLES
# =========================================================================

class FakeResponse:
    def __init__(self, payload=None, status_code=200, raw=None):
        self.status_code = status_code
        self._payload = payload
        self.content = (raw if raw is not None
                        else json.dumps(payload or {}).encode("utf-8"))

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def geojson(coords):
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature",
                          "properties": {"summary": {"distance": 999.0,
                                                     "duration": 888.0}},
                          "geometry": {"type": "LineString",
                                       "coordinates": coords}}]}


class Directions:
    """Remplace requests.post et enregistre chaque appel."""

    def __init__(self, responses=None, exception=None):
        self.calls = []
        self.responses = responses
        self.exception = exception

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers,
                           "timeout": timeout})
        if self.exception is not None:
            raise self.exception
        if self.responses is None:
            trace = [[2.0, 48.0], [2.1, 48.1], [2.2, 48.2]]
            return FakeResponse(geojson(trace))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return FakeResponse(**self.responses[index])


def route(n=3, base=0.0):
    return [[2.0 + base + i * 0.01, 48.0 + base + i * 0.01] for i in range(n)]


class MapGeometryTestCase(unittest.TestCase):
    """Appelle la vue directement : flask est un double, il n'y a pas de
    serveur. request et jsonify sont remplaces le temps du test."""

    def setUp(self):
        app._MAP_GEOMETRY_CACHE.clear()
        app._reset_map_stats()
        self._key = app.ORS_KEY
        app.ORS_KEY = "cle-de-test"
        self._post = app.requests.post
        self.addCleanup(self._restore)

    def _restore(self):
        app.ORS_KEY = self._key
        app.requests.post = self._post
        app._MAP_GEOMETRY_CACHE.clear()
        app._reset_map_stats()

    def call(self, body, directions=None, raw=None):
        self.directions = directions or Directions()
        app.requests.post = self.directions

        payload = raw if raw is not None else json.dumps(body).encode("utf-8")
        captured = {}

        class _Request:
            @staticmethod
            def get_data(cache=False, as_text=False):
                return payload

        def _jsonify(obj):
            captured["body"] = obj
            return obj

        real_request, real_jsonify = app.request, app.jsonify
        app.request, app.jsonify = _Request, _jsonify
        try:
            _, status = app.map_geometry()
        finally:
            app.request, app.jsonify = real_request, real_jsonify
        return captured["body"], status


# =========================================================================
# VALIDATION
# =========================================================================

class TestValidation(MapGeometryTestCase):

    def test_two_valid_routes_are_accepted(self):
        body, status = self.call({"routes": [route(), route(3, 1.0)],
                                  "profile": "driving-car"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(len(body["geometries"]), 2)
        self.assertFalse(body["fallback_used"])

    def test_the_profile_defaults_to_the_only_allowed_one(self):
        body, status = self.call({"routes": [route(), route(3, 1.0)]})
        self.assertEqual(status, 200)
        self.assertIn("driving-car", self.directions.calls[0]["url"])

    def test_a_single_route_is_refused(self):
        body, status = self.call({"routes": [route()]})
        self.assertEqual(status, 400)
        self.assertEqual(body["status"], "validation_error")
        self.assertIsNone(body["geometries"])
        self.assertEqual(self.directions.calls, [])

    def test_three_routes_are_refused(self):
        body, status = self.call({"routes": [route(), route(3, 1.0), route(3, 2.0)]})
        self.assertEqual(status, 400)
        self.assertEqual(self.directions.calls, [])

    def test_more_than_sixty_coordinates_is_refused(self):
        body, status = self.call({"routes": [route(61), route(3, 1.0)]})
        self.assertEqual(status, 400)
        self.assertIn("exceeds 60", body["error"])
        self.assertEqual(self.directions.calls, [])

    def test_exactly_sixty_coordinates_is_accepted(self):
        body, status = self.call({"routes": [route(60), route(3, 1.0)]})
        self.assertEqual(status, 200)

    def test_a_single_coordinate_route_is_refused(self):
        body, status = self.call({"routes": [route(1), route(3, 1.0)]})
        self.assertEqual(status, 400)
        self.assertIn("at least 2", body["error"])

    def test_invalid_coordinates_are_refused(self):
        for bad in ([["a", 48.0], [2.0, 48.0]],
                    [[2.0], [2.0, 48.0]],
                    [[200.0, 48.0], [2.0, 48.0]],
                    [[2.0, 91.0], [2.0, 48.0]],
                    [[float("inf"), 48.0], [2.0, 48.0]],
                    [[True, 48.0], [2.0, 48.0]]):
            body, status = self.call({"routes": [bad, route(3, 1.0)]})
            self.assertEqual(status, 400, "coordonnee acceptee a tort: %r" % bad)
            self.assertEqual(self.directions.calls, [])

    def test_a_foreign_profile_is_refused_never_substituted(self):
        for bad in ("cycling-regular", "foot-walking", "driving-hgv", "", 42):
            body, status = self.call({"routes": [route(), route(3, 1.0)],
                                      "profile": bad})
            self.assertEqual(status, 400, "profil accepte a tort: %r" % bad)
            self.assertEqual(self.directions.calls, [])

    def test_no_client_supplied_url_is_ever_used(self):
        self.call({"routes": [route(), route(3, 1.0)],
                   "url": "https://exemple.invalide/proxy",
                   "host": "exemple.invalide"})
        for call in self.directions.calls:
            self.assertTrue(call["url"].startswith(
                "https://api.heigit.org/openrouteservice/v2/directions/"))
            self.assertNotIn("exemple.invalide", call["url"])

    def test_an_oversized_body_is_refused_before_parsing(self):
        raw = b"x" * (app.MAP_GEOMETRY_MAX_BODY_BYTES + 1)
        body, status = self.call({}, raw=raw)
        self.assertEqual(status, 413)
        self.assertEqual(body["status"], "body_too_large")
        self.assertEqual(self.directions.calls, [])

    def test_malformed_json_is_refused(self):
        body, status = self.call({}, raw=b"{ pas du json")
        self.assertEqual(status, 400)
        self.assertEqual(body["status"], "validation_error")


# =========================================================================
# APPELS ET CACHE
# =========================================================================

class TestCallsAndCache(MapGeometryTestCase):

    def test_at_most_two_directions_calls(self):
        body, _ = self.call({"routes": [route(), route(3, 1.0)]})
        self.assertEqual(len(self.directions.calls), 2)
        self.assertEqual(body["calls"], 2)

    def test_a_second_identical_request_costs_nothing(self):
        first, _ = self.call({"routes": [route(), route(3, 1.0)]})
        self.assertEqual(first["calls"], 2)
        self.assertFalse(first["cache_hit"])

        second, _ = self.call({"routes": [route(), route(3, 1.0)]})
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["calls"], 0)
        self.assertEqual(second["status"], "cached")
        self.assertEqual(second["geometries"], first["geometries"])
        self.assertEqual(self.directions.calls, [])

    def test_a_different_order_is_a_different_key(self):
        self.call({"routes": [route(), route(3, 1.0)]})
        reversed_first = list(reversed(route()))
        body, _ = self.call({"routes": [reversed_first, route(3, 1.0)]})
        self.assertFalse(body["cache_hit"])
        self.assertEqual(len(self.directions.calls), 2)

    def test_the_cache_is_bounded(self):
        for i in range(app.MAP_GEOMETRY_CACHE_MAX + 4):
            self.call({"routes": [route(3, i * 0.5), route(3, i * 0.5 + 0.1)]})
        self.assertLessEqual(len(app._MAP_GEOMETRY_CACHE),
                             app.MAP_GEOMETRY_CACHE_MAX)

    def test_map_calls_never_touch_the_optimisation_counters(self):
        app._reset_api_stats()
        self.call({"routes": [route(), route(3, 1.0)]})
        self.assertEqual(app._API_STATS["vroom"], 0)
        self.assertEqual(app._API_STATS["matrix"], 0)
        self.assertEqual(app._api_calls_total(), 0)
        self.assertEqual(app._MAP_STATS["directions"], 2)

    def test_the_map_cache_is_separate_from_the_matrix_cache(self):
        app._MATRIX_CACHE.clear()
        self.call({"routes": [route(), route(3, 1.0)]})
        self.assertEqual(app._MATRIX_CACHE, {})
        self.assertTrue(app._MAP_GEOMETRY_CACHE)


# =========================================================================
# ECHECS
# =========================================================================

class TestFailures(MapGeometryTestCase):

    def test_a_missing_ors_key_is_reported_not_crashed(self):
        app.ORS_KEY = ""
        body, status = self.call({"routes": [route(), route(3, 1.0)]})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "missing_ors_key")
        self.assertIsNone(body["geometries"])
        self.assertTrue(body["fallback_used"])
        self.assertEqual(self.directions.calls, [])

    def test_a_timeout_falls_back_cleanly(self):
        class _Timeout(Exception):
            """Nom volontaire : le repli classe l'exception par son nom, pour
            ne pas dependre d'une classe absente quand requests est double."""

        directions = Directions(exception=_Timeout())
        body, status = self.call({"routes": [route(), route(3, 1.0)]},
                                 directions=directions)
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "timeout")
        self.assertTrue(body["fallback_used"])
        self.assertIsNone(body["geometries"])

    def test_an_ors_error_falls_back_cleanly(self):
        directions = Directions(responses=[{"payload": {"error": "quota"},
                                            "status_code": 429}])
        body, status = self.call({"routes": [route(), route(3, 1.0)]},
                                 directions=directions)
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "http_429")
        self.assertTrue(body["fallback_used"])

    def test_an_unreadable_response_falls_back_cleanly(self):
        directions = Directions(responses=[{"payload": None, "raw": b"<html>"}])
        body, status = self.call({"routes": [route(), route(3, 1.0)]},
                                 directions=directions)
        self.assertEqual(body["status"], "invalid_response")
        self.assertTrue(body["fallback_used"])

    def test_an_oversized_response_is_refused(self):
        huge = b"x" * (app.MAP_GEOMETRY_MAX_RESPONSE_BYTES + 1)
        directions = Directions(responses=[{"payload": {}, "raw": huge}])
        body, status = self.call({"routes": [route(), route(3, 1.0)]},
                                 directions=directions)
        self.assertEqual(body["status"], "response_too_large")
        self.assertTrue(body["fallback_used"])

    def test_a_partial_failure_keeps_the_route_that_worked(self):
        trace = [[2.0, 48.0], [2.1, 48.1]]
        directions = Directions(responses=[
            {"payload": geojson(trace)},
            {"payload": {"error": "boom"}, "status_code": 500},
        ])
        body, status = self.call({"routes": [route(), route(3, 1.0)]},
                                 directions=directions)
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "partial")
        self.assertIsNotNone(body["geometries"][0])
        self.assertIsNone(body["geometries"][1])
        self.assertTrue(body["fallback_used"])

    def test_a_partial_result_is_never_cached(self):
        directions = Directions(responses=[
            {"payload": geojson([[2.0, 48.0], [2.1, 48.1]])},
            {"payload": {"error": "boom"}, "status_code": 500},
        ])
        self.call({"routes": [route(), route(3, 1.0)]}, directions=directions)
        self.assertEqual(app._MAP_GEOMETRY_CACHE, {})

    def test_no_secret_ever_appears_in_the_response(self):
        app.ORS_KEY = "SECRET-ORS-123"
        for scenario in ({"routes": [route(), route(3, 1.0)]},
                         {"routes": [route()]},
                         {"routes": [route(), route(3, 1.0)], "profile": "bad"}):
            body, _ = self.call(scenario)
            self.assertNotIn("SECRET-ORS-123", json.dumps(body))


# =========================================================================
# GEOMETRIE RENVOYEE
# =========================================================================

class TestGeometryPayload(MapGeometryTestCase):

    def test_the_coordinate_order_is_preserved(self):
        trace = [[2.0, 48.0], [2.5, 48.5], [3.0, 49.0]]
        directions = Directions(responses=[{"payload": geojson(trace)}])
        body, _ = self.call({"routes": [route(), route(3, 1.0)]},
                            directions=directions)
        self.assertEqual(body["geometries"][0], trace)
        self.assertEqual(body["geometries"][0][0], [2.0, 48.0])
        self.assertEqual(body["geometries"][0][-1], [3.0, 49.0])

    def test_directions_distances_and_durations_are_discarded(self):
        """Les metriques du Benchmark viennent de la matrice ORS. Celles de
        Directions ne doivent jamais s'y substituer, meme par inadvertance."""
        body, _ = self.call({"routes": [route(), route(3, 1.0)]})
        serialised = json.dumps(body)
        self.assertNotIn("999", serialised)
        self.assertNotIn("888", serialised)
        self.assertNotIn("distance", serialised)
        self.assertNotIn("duration", serialised)

    def test_the_requested_order_is_sent_untouched_to_ors(self):
        first, second = route(), route(3, 1.0)
        self.call({"routes": [first, second]})
        self.assertEqual(self.directions.calls[0]["json"]["coordinates"], first)
        self.assertEqual(self.directions.calls[1]["json"]["coordinates"], second)

    def test_the_network_timeout_is_bounded(self):
        self.call({"routes": [route(), route(3, 1.0)]})
        for call in self.directions.calls:
            self.assertEqual(call["timeout"], app.MAP_GEOMETRY_TIMEOUT_S)
            self.assertLessEqual(call["timeout"], 30)


# =========================================================================
# ISOLEMENT VIS-A-VIS DE L'OPTIMISATION
# =========================================================================

class TestOptimisationIsolation(unittest.TestCase):

    def test_optimize_never_calls_the_geometry_layer(self):
        """Preuve statique : aucune fonction d'optimisation ne reference la
        couche geometrie. Le cout de la carte ne peut pas fuir dans le
        temps ni dans les appels mesures du Benchmark."""
        import ast
        with open("app.py", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)

        geometry_symbols = ("_post_directions", "_fetch_route_geometry",
                            "map_geometry", "_map_geometry_cache_get",
                            "_map_geometry_cache_put", "ORS_DIRECTIONS_URL",
                            "_MAP_STATS", "_MAP_GEOMETRY_CACHE")
        allowed = {"map_geometry", "_reset_map_stats", "_post_directions",
                   "_fetch_route_geometry", "_map_geometry_cache_key",
                   "_map_geometry_cache_get", "_map_geometry_cache_put",
                   "_validate_map_geometry", "_extract_geometry"}

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name in allowed:
                continue
            body = ast.get_source_segment(source, node) or ""
            for symbol in geometry_symbols:
                if symbol in body:
                    offenders.append("%s reference %s" % (node.name, symbol))
        self.assertEqual(offenders, [])

    def test_the_geometry_layer_never_touches_the_optimisation_counters(self):
        import ast
        with open("app.py", encoding="utf-8") as handle:
            source = handle.read()
        tree = ast.parse(source)
        geometry_functions = {"map_geometry", "_post_directions",
                              "_fetch_route_geometry", "_validate_map_geometry",
                              "_extract_geometry", "_map_geometry_cache_get",
                              "_map_geometry_cache_put"}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in geometry_functions:
                # Le CODE seul : une mention en commentaire ou en docstring
                # explique justement pourquoi la separation existe, elle ne
                # la viole pas.
                stripped = ast.parse(ast.get_source_segment(source, node))
                for sub in ast.walk(stripped):
                    if (isinstance(sub, ast.Expr)
                            and isinstance(sub.value, ast.Constant)
                            and isinstance(sub.value.value, str)):
                        sub.value.value = ""
                body = ast.unparse(stripped)
                for forbidden in ("_API_STATS", "_post_matrix", "_post_vroom",
                                  "_MATRIX_CACHE"):
                    self.assertNotIn(forbidden, body,
                                     "%s touche %s" % (node.name, forbidden))

    def test_the_optimisation_routes_are_unchanged(self):
        import re
        with open("app.py", encoding="utf-8") as handle:
            source = handle.read()
        routes = re.findall(r'@app\.route\("([^"]+)"', source)
        self.assertEqual(sorted(routes),
                         ["/", "/healthz", "/map-geometry", "/optimize"])

    def test_the_outbound_optimisation_calls_are_unchanged(self):
        with open("app.py", encoding="utf-8") as handle:
            source = handle.read()
        # Les deux points de sortie historiques de l'optimisation restent
        # les deux seuls ; Directions a le sien, distinct et compte a part.
        self.assertEqual(source.count("requests.post(ORS_VROOM_URL"), 1)
        self.assertEqual(source.count("requests.post(ORS_MATRIX_URL"), 1)
        self.assertEqual(source.count("_MAP_STATS[\"directions\"] += 1"), 1)


if __name__ == "__main__":
    unittest.main()
