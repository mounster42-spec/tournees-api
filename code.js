// =========================
// CONSTANTES
// =========================
const API_BASE = "https://tournees-api.onrender.com";

const STRATEGIES = ["kmeans", "ortools_haversine", "ortools_ors_matrix",
                    "ortools_ors_matrix_connected",
                    "hybrid_local_vroom_territorial"];
const DEFAULT_STRATEGY = "kmeans";

// Identifiant backend de la stratégie expérimentale. Le libellé de menu et
// cet identifiant sont liés en UN SEUL endroit : une entrée de menu qui
// enverrait un autre nom produirait un 501 côté API, et surtout une ligne de
// Benchmark étiquetée avec une stratégie qui n'a pas tourné.
const EXP_STRATEGY = "hybrid_local_vroom_territorial";
const EXP_STRATEGY_LABEL = "[EXP] VROOM local + ALNS territoriale";

// Ligne de la feuille "Paramètres" portant la stratégie
const STRATEGY_ROW = 6;

// Libelles du champ "Mode" de la feuille Résultats.
// Indexes par partition_engine, qui est la source la plus precise : sous
// strategy_used = kmeans, il distingue l'affectation faite par Vroom multi
// de celle faite par K-Means.
// Vroom séquence dans les quatre cas ; seul le moteur d'affectation change.
const ENGINE_LABELS = {
  "vroom_multi":        "Vroom (affectation + séquencement)",
  "kmeans_fallback":    "K-Means (affectation) + Vroom (séquencement)",
  "ortools_haversine":  "OR-Tools Haversine (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix": "OR-Tools ORS Matrix (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix_connected":
    "OR-Tools ORS Matrix — territoires connexes (affectation) + Vroom (séquencement)",
  "hybrid_local_vroom_territorial":
    "[EXP] VROOM local conjoint + ALNS territoriale (affectation + séquencement)"
};

// Repli si partition_engine est absent (backend antérieur au lot 3).
const STRATEGY_LABELS = {
  "kmeans":             "K-Means (affectation) + Vroom (séquencement)",
  "ortools_haversine":  "OR-Tools Haversine (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix": "OR-Tools ORS Matrix (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix_connected":
    "OR-Tools ORS Matrix — territoires connexes (affectation) + Vroom (séquencement)",
  "hybrid_local_vroom_territorial":
    "[EXP] VROOM local conjoint + ALNS territoriale (affectation + séquencement)"
};

const BENCH_SHEET = "Benchmark";

// Colonnes historiques : ordre et libellés figés, ne jamais déplacer ni renommer.
const BENCH_HEADERS_BASE = [
  "Date", "Stratégie exécutée", "Stratégie demandée", "Nb pts", "Signature jeu", "Nb véh",
  "Km T1", "Km T2", "Km total",
  "Min T1", "Min T2", "Min total",
  "Temps calcul (s)", "Appels API", "Vroom", "Matrix",
  "optimization_path", "Répartition"
];

// Colonnes ajoutées par le lot D-3, strictement à la fin.
const BENCH_HEADERS_D3 = [
  "d3_label",
  "max_swap_candidates", "swap_max_consecutive_fails",
  "swap_candidates_tested", "swaps_accepted",
  "swap_resequence_cache_hits", "swap_resequence_cache_misses",
  "swap_vroom_calls_saved", "swap_stop_reason",
  "ortools_solution_limit_effective", "run_error"
];

// Colonnes du certificat territorial, strictement à la fin.
const BENCH_HEADERS_TERR = [
  "territorial_partition", "territorial_method",
  "territorial_membership_locked", "territorial_side_violations",
  "territorial_separator_angle_deg", "territorial_separator_margin_m",
  "territorial_overlap_status", "territorial_candidates_unique",
  "territorial_candidates_scored", "territorial_fallback_used",
  "territorial_error"
];

// Colonnes du certificat de connexite, strictement a la fin.
const BENCH_HEADERS_CONN = [
  "connected_partition", "connected_method", "connected_membership_locked",
  "connected_target_sizes", "connected_components_t1", "connected_components_t2",
  "connected_component_sizes_t1", "connected_component_sizes_t2",
  "connected_candidates_generated", "connected_candidates_valid",
  "connected_candidates_scored", "connected_candidates_ortools",
  "connected_candidates_vroom", "connected_cut_edges", "connected_cut_length_m",
  "connected_cross_neighbors", "connected_enclave_points",
  "connected_selected_seed", "connected_fallback_used", "connected_error",
  "connected_vroom_calls", "selected_sequencer", "final_selection_reason",
  "ortools_total_duration_s", "ortools_total_distance_m",
  "vroom_total_duration_s", "vroom_total_distance_m", "partition_solver"
];

// Colonnes du recentrage ORS-first, strictement a la fin : reference a
// cardinalite exacte, penalite de connexite, budgets par etape, source
// ORS-first et arbitrage de la post-optimisation. Aucune colonne existante
// n'est deplacee.
const BENCH_HEADERS_ORSFIRST = [
  "connected_ors_reference_available", "connected_ors_reference_duration_s",
  "connected_ors_reference_distance_m", "connected_ors_reference_sizes",
  "connected_ors_reference_components_t1",
  "connected_ors_reference_components_t2",
  "connected_ors_reference_time_limit_hit",
  "connected_ors_reference_fallback_used",
  "connected_ors_reference_solver_status", "connected_ors_reference_solve_ms",
  "connectivity_penalty_duration_s", "connectivity_penalty_distance_m",
  "connectivity_penalty_reliable", "connectivity_penalty_note",
  "connected_stage_timings_text", "connected_stage_budget_exhausted",
  "connected_generation_expired_after",
  "connected_ors_repair_candidates_raw",
  "connected_ors_repair_candidates_unique",
  "connected_ors_repair_reached_ortools",
  "connected_ors_repair_best_proxy_rank", "connected_ors_repair_is_winner",
  "connected_prescore_refined", "connected_prescore_budget_exhausted",
  "connected_winner_proxy_rank_rough", "connected_winner_proxy_rank_refined",
  "connected_winner_ortools_rank",
  "connected_matrix_hash", "connected_per_source_text",
  "post_optimization_kept", "post_optimization_note"
];

// Colonnes de la stratégie expérimentale VROOM local + ALNS territoriale,
// strictement à la fin. Aucune colonne existante n'est déplacée, renommée ni
// supprimée : ces colonnes restent vides pour les quatre stratégies de
// production, qui ne renvoient pas ce bloc de diagnostic.
//
// Toutes ces valeurs proviennent de result.ors_matrix.hybrid, déjà présent
// dans la réponse : aucune route Flask n'a eu besoin de changer.
const BENCH_HEADERS_HYBRID = [
  // contexte du run expérimental
  "hybrid_error", "local_vroom_enabled", "local_vroom_version",
  // juge commun : la seule mesure qui classe les solutions
  "common_rescore_duration_s", "common_rescore_distance_m",
  "common_rescore_matrix_hash",
  // résultat de chaque bloc
  "joint_direct_valid", "joint_direct_duration_s", "joint_direct_sizes",
  "joint_nucleus_attempted", "joint_nucleus_valid",
  "joint_nucleus_best_duration_s",
  "route_first_unique", "route_first_best_duration_s",
  "joint_alns_iterations", "joint_alns_accepted", "joint_alns_seed",
  "joint_alns_best_duration_s",
  "joint_finalists", "joint_finalists_local_vroom_solved",
  "joint_finalists_reused",
  // sélection finale
  "joint_solutions_considered", "joint_selected_source",
  "joint_selected_duration_s", "joint_selected_distance_m",
  "joint_selected_sizes", "joint_selected_components",
  "joint_selected_enclaves",
  // admissibilité territoriale
  "joint_territorial_level", "joint_territorial_max_enclaves",
  "joint_territorial_admissible", "joint_territorial_fallback_used",
  "joint_territorial_fallback_reason", "joint_territorial_thresholds",
  "joint_territorial_level_counts",
  // compteur de résolutions locales
  "local_vroom_solve_count", "local_vroom_attempted", "local_vroom_succeeded",
  "local_vroom_failed", "local_vroom_timed_out", "local_vroom_reused",
  "local_vroom_skipped_for_time", "local_vroom_elapsed_ms",
  "local_vroom_stop_reason", "local_vroom_last_error",
  // discipline temporelle, un temps par bloc réellement renvoyé
  "hybrid_stage_matrix_ms", "hybrid_stage_joint_direct_ms",
  "hybrid_stage_route_first_ms", "hybrid_stage_joint_nucleus_ms",
  "hybrid_stage_joint_alns_ms", "hybrid_stage_alns_refine_ms",
  "hybrid_stage_joint_finalists_ms",
  "hybrid_stage_timings_text", "hybrid_stage_stops_text",
  "hybrid_total_elapsed_ms", "hybrid_soft_limit_reached"
];

const BENCH_HEADERS = BENCH_HEADERS_BASE
  .concat(BENCH_HEADERS_D3)
  .concat(BENCH_HEADERS_TERR)
  .concat(BENCH_HEADERS_CONN)
  .concat(BENCH_HEADERS_ORSFIRST)
  .concat(BENCH_HEADERS_HYBRID);



function setupSheets() {

  const ss = SpreadsheetApp.getActive();

  // --- Feuille Paramètres ---
  let paramSheet = ss.getSheetByName("Paramètres");
  if (!paramSheet) {
    paramSheet = ss.insertSheet("Paramètres");
  }
  paramSheet.clear();

  const paramHeaders = ["Paramètre", "Valeur"];
  const paramData = [
    paramHeaders,
    ["Nombre de véhicules", 2],
    ["Max points par véhicule", 35],
    ["Point de départ (ID)", ""],
    ["Point d'arrivée (ID)", ""],
    ["Stratégie", DEFAULT_STRATEGY]
  ];

  paramSheet.getRange(1, 1, paramData.length, 2).setValues(paramData);

  // Menu déroulant sur la cellule Stratégie
  paramSheet.getRange(STRATEGY_ROW, 2).setDataValidation(strategyValidationRule());

  // Style header
  const paramHeaderRange = paramSheet.getRange(1, 1, 1, 2);
  paramHeaderRange.setBackground("#4a86c8").setFontColor("#ffffff").setFontWeight("bold");
  paramSheet.setColumnWidth(1, 250);
  paramSheet.setColumnWidth(2, 200);

  // Bordures
  paramSheet.getRange(1, 1, paramData.length, 2)
    .setBorder(true, true, true, true, true, true);

  // --- Feuille Horodateurs ---
  let horoSheet = ss.getSheetByName("Horodateurs");
  if (!horoSheet) {
    horoSheet = ss.insertSheet("Horodateurs");
  }

  // Ne clear que si vide (pour ne pas écraser les données existantes)
  if (horoSheet.getLastRow() <= 1) {
    horoSheet.clear();
    const horoHeaders = ["ID", "Adresse", "Latitude", "Longitude", "Sélection"];
    horoSheet.getRange(1, 1, 1, 5).setValues([horoHeaders]);
  }

  const horoHeaderRange = horoSheet.getRange(1, 1, 1, 5);
  horoHeaderRange.setBackground("#6aa84f").setFontColor("#ffffff").setFontWeight("bold");
  horoSheet.setColumnWidth(1, 100);
  horoSheet.setColumnWidth(2, 300);
  horoSheet.setColumnWidth(3, 120);
  horoSheet.setColumnWidth(4, 120);
  horoSheet.setColumnWidth(5, 100);

  // Checkbox dans la colonne Sélection (lignes 2 à 100)
  horoSheet.getRange(2, 5, 99, 1).insertCheckboxes();

  // --- Feuille Résultats ---
  let resSheet = ss.getSheetByName("Résultats");
  if (!resSheet) {
    resSheet = ss.insertSheet("Résultats");
  }
  resSheet.clear();

  const resHeaderRange = resSheet.getRange(1, 1, 1, 1);
  resHeaderRange.setValue("Résultats de l'optimisation");
  resHeaderRange.setBackground("#e06666").setFontColor("#ffffff").setFontWeight("bold");
  resSheet.setColumnWidth(1, 150);
  resSheet.setColumnWidth(2, 600);

  // NB : la feuille "Benchmark" n'est jamais touchée ici, son historique est conservé.

  SpreadsheetApp.getActive().toast("Feuilles créées et mises en page !", "Setup", 3);
}


// =========================
// STRATÉGIE : VALIDATION + MIGRATION
// =========================
function strategyValidationRule() {
  return SpreadsheetApp.newDataValidation()
    .requireValueInList(STRATEGIES, true)
    .setAllowInvalid(false)
    .build();
}


/**
 * Ajoute la ligne "Stratégie" aux feuilles Paramètres déjà existantes.
 * Évite d'avoir à relancer setupSheets(), qui efface Paramètres et Résultats.
 * Idempotent : n'écrase jamais une valeur déjà saisie.
 */
function ensureStrategyCell() {

  const sheet = SpreadsheetApp.getActive().getSheetByName("Paramètres");
  if (!sheet) return;

  if (String(sheet.getRange(STRATEGY_ROW, 1).getValue()).trim() === "") {
    sheet.getRange(STRATEGY_ROW, 1).setValue("Stratégie");
    sheet.getRange(STRATEGY_ROW, 2).setValue(DEFAULT_STRATEGY);
    sheet.getRange(STRATEGY_ROW, 1, 1, 2)
      .setBorder(true, true, true, true, true, true);
  }

  sheet.getRange(STRATEGY_ROW, 2).setDataValidation(strategyValidationRule());
}


// =========================
// LIRE PARAMÈTRES
// =========================
function getParams() {

  const sheet = SpreadsheetApp.getActive().getSheetByName("Paramètres");
  const data = sheet.getRange(2, 2, 5, 1).getValues();

  // Cellule vide (feuille antérieure au lot 2) -> défaut kmeans.
  // Valeur saisie mais inconnue -> erreur explicite : une ligne de Benchmark
  // ne doit jamais porter une étiquette de stratégie fausse.
  const raw = data[4][0] === "" || data[4][0] === null || data[4][0] === undefined
    ? DEFAULT_STRATEGY
    : String(data[4][0]).trim().toLowerCase();

  if (STRATEGIES.indexOf(raw) === -1) {
    throw new Error(
      "Stratégie inconnue en Paramètres!B" + STRATEGY_ROW + " : \"" + raw + "\". " +
      "Valeurs acceptées : " + STRATEGIES.join(", ")
    );
  }

  return {
    num_vehicles: Number(data[0][0]) || 2,
    max_per_vehicle: Number(data[1][0]) || 35,
    start_id: data[2][0] ? String(data[2][0]) : "",
    end_id: data[3][0] ? String(data[3][0]) : "",
    strategy: raw
  };
}


// =========================
// LIRE POINTS
// =========================
function getPoints() {

  const sheet = SpreadsheetApp.getActive().getSheetByName("Horodateurs");
  const data = sheet.getDataRange().getValues();

  let points = [];

  for (let i = 1; i < data.length; i++) {

    const selection = data[i][4];

    if (selection === true || selection === "TRUE") {

      points.push({
        id: String(data[i][0]),
        address: data[i][1],
        lat: Number(data[i][2]),
        lon: Number(data[i][3])
      });

    }
  }

  return points;
}


// =========================
// APPEL API
// =========================
function callAPI(points, params) {

  const url = API_BASE + "/optimize";

  const payload = {
    points: points,
    num_vehicles: params.num_vehicles,
    max_per_vehicle: params.max_per_vehicle,
    start_id: params.start_id,
    end_id: params.end_id,
    strategy: params.strategy
  };

  // Réveil du serveur (Render free tier s'endort, peut prendre 30-60s)
  for (var attempt = 0; attempt < 6; attempt++) {
    try {
      var wakeResp = UrlFetchApp.fetch(API_BASE + "/", { muteHttpExceptions: true });
      if (wakeResp.getResponseCode() === 200) break;
    } catch (e) {}
    Utilities.sleep(10000);
  }

  const response = UrlFetchApp.fetch(url, {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true
  });

  const text = response.getContentText();
  const code = response.getResponseCode();

  // Le backend renvoie un corps JSON explicite sur 400 (stratégie inconnue)
  // et 501 (stratégie pas encore implémentée). Sans ce traitement, ces cas
  // étaient masqués par le message générique "L'API n'est pas prête".
  if (code !== 200) {
    var detail = "";
    try {
      var errJson = JSON.parse(text);
      if (errJson && errJson.error) detail = String(errJson.error);
    } catch (e) {}

    if (detail) {
      throw new Error("API (code " + code + ") : " + detail);
    }
    throw new Error("L'API n'est pas prête (code " + code + "). Réessayez dans 1 minute.");
  }

  if (text.startsWith("<!")) {
    throw new Error("L'API n'est pas prête (réponse HTML). Réessayez dans 1 minute.");
  }

  return JSON.parse(text);
}


// =========================
// LIBELLÉ DU MODE
// =========================
/**
 * Construit le texte de la cellule "Mode" de la feuille Résultats.
 *
 * L'ancienne version se déduisait uniquement de vroom_used, ce qui produisait
 * deux libellés faux depuis l'arrivée du sélecteur de stratégie :
 *   - vroom_used = false affichait "K-Means (affectation)" quelle que soit la
 *     stratégie, y compris sous ortools_haversine / ortools_ors_matrix ;
 *   - vroom_used = true affichait "Vroom (affectation + séquencement)" alors
 *     que sur 62 points l'affectation vient de K-Means.
 *
 * Le libellé ne suppose donc plus rien : il vient de partition_engine, sinon
 * de strategy_used, et reste explicitement indéterminé si aucun des deux n'est
 * renseigné. Un échec de séquencement Vroom est signalé à part, car il ne
 * change jamais la stratégie d'affectation.
 */
function buildModeText(result) {

  var label = ENGINE_LABELS[result.partition_engine]
           || STRATEGY_LABELS[result.strategy_used]
           || "Mode indéterminé (réponse sans partition_engine ni strategy_used)";

  var text = label;

  var steps = result.post_processing;
  if (steps && steps.length) {
    text += " | post-traitement : " + steps.join(", ");
  }

  // === et non !result.vroom_used : une réponse sans le champ ne doit pas
  // être interprétée comme un échec de séquencement.
  if (result.vroom_used === false) {
    text += " | fallback séquencement : " + (result.vroom_error || "raison inconnue");
  }

  return text;
}


// =========================
// ÉCRIRE RÉSULTATS
// =========================
function writeResult(result, params, points) {

  const ss = SpreadsheetApp.getActive();
  let sheet = ss.getSheetByName("Résultats");
  if (!sheet) {
    sheet = ss.insertSheet("Résultats");
  }
  sheet.clear();

  // Dictionnaires ID → Adresse / Lat / Lon
  const addressMap = {};
  const latMap = {};
  const lonMap = {};
  for (let i = 0; i < points.length; i++) {
    const sid = String(points[i].id);
    addressMap[sid] = points[i].address || "";
    latMap[sid] = points[i].lat || "";
    lonMap[sid] = points[i].lon || "";
  }

  // Récupérer les tournées
  const t1 = result["tournee_1"] || [];
  const t2 = result["tournee_2"] || [];
  const maxLen = Math.max(t1.length, t2.length);

  // Largeurs colonnes
  sheet.setColumnWidth(1, 70);   // Ordre
  sheet.setColumnWidth(2, 70);   // T1 ID
  sheet.setColumnWidth(3, 300);  // T1 Adresse
  sheet.setColumnWidth(4, 100);  // T1 Lat
  sheet.setColumnWidth(5, 100);  // T1 Lon
  sheet.setColumnWidth(6, 70);   // T2 ID
  sheet.setColumnWidth(7, 300);  // T2 Adresse
  sheet.setColumnWidth(8, 100);  // T2 Lat
  sheet.setColumnWidth(9, 100);  // T2 Lon

  // --- Ligne 1 : en-têtes principaux ---
  sheet.getRange(1, 1).setValue("Ordre");
  sheet.getRange(1, 1).setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");

  sheet.getRange(1, 2, 1, 4).merge();
  sheet.getRange(1, 2).setValue("Tournée 1 (" + t1.length + " pts)");
  sheet.getRange(1, 2).setBackground("#6aa84f").setFontColor("#ffffff").setFontWeight("bold");
  sheet.getRange(1, 2).setHorizontalAlignment("center");

  sheet.getRange(1, 6, 1, 4).merge();
  sheet.getRange(1, 6).setValue("Tournée 2 (" + t2.length + " pts)");
  sheet.getRange(1, 6).setBackground("#4a86c8").setFontColor("#ffffff").setFontWeight("bold");
  sheet.getRange(1, 6).setHorizontalAlignment("center");

  // --- Ligne 2 : sous-en-têtes ---
  sheet.getRange(2, 1, 1, 9).setValues([["", "ID", "Adresse", "Lat", "Lon", "ID", "Adresse", "Lat", "Lon"]]);
  sheet.getRange(2, 1, 1, 9).setBackground("#f3f3f3").setFontWeight("bold");

  // --- Lignes de données ---
  if (maxLen > 0) {
    const rows = [];
    for (let i = 0; i < maxLen; i++) {
      const id1 = i < t1.length ? String(t1[i]) : "";
      const addr1 = id1 ? (addressMap[id1] || "") : "";
      const lat1 = id1 ? (latMap[id1] || "") : "";
      const lon1 = id1 ? (lonMap[id1] || "") : "";
      const id2 = i < t2.length ? String(t2[i]) : "";
      const addr2 = id2 ? (addressMap[id2] || "") : "";
      const lat2 = id2 ? (latMap[id2] || "") : "";
      const lon2 = id2 ? (lonMap[id2] || "") : "";
      rows.push([i + 1, id1, addr1, lat1, lon1, id2, addr2, lat2, lon2]);
    }
    sheet.getRange(3, 1, rows.length, 9).setValues(rows);

    // Alternance couleur lignes
    for (let i = 0; i < rows.length; i++) {
      const bg = (i % 2 === 0) ? "#ffffff" : "#f9f9f9";
      sheet.getRange(3 + i, 1, 1, 9).setBackground(bg);
    }
  }

  // Bordures sur tout le tableau
  const totalRows = maxLen + 2;
  sheet.getRange(1, 1, totalRows, 9).setBorder(true, true, true, true, true, true);

  // Séparateur visuel entre T1 et T2
  sheet.getRange(1, 6, totalRows, 1)
    .setBorder(null, true, null, null, null, null, "#000000", SpreadsheetApp.BorderStyle.SOLID_MEDIUM);

  // Info clusters + mode + distances
  const infoRow = totalRows + 2;
  sheet.getRange(infoRow, 1).setValue("Clusters DBSCAN");
  sheet.getRange(infoRow, 2).setValue(result.num_clusters_dbscan || "");
  sheet.getRange(infoRow, 1).setFontWeight("bold");

  var modeText = buildModeText(result);
  sheet.getRange(infoRow + 1, 1).setValue("Mode");
  sheet.getRange(infoRow + 1, 2).setValue(modeText);
  sheet.getRange(infoRow + 1, 1).setFontWeight("bold");

  var km1  = result.tournee_1_km  || 0;
  var km2  = result.tournee_2_km  || 0;
  var min1 = result.tournee_1_min;
  var min2 = result.tournee_2_min;

  var dist1Text = km1 + " km routiers" + (min1 != null ? " (~" + min1 + " min)" : "");
  var dist2Text = km2 + " km routiers" + (min2 != null ? " (~" + min2 + " min)" : "");
  var totalKm   = Math.round((km1 + km2) * 100) / 100;
  var totalMin  = (min1 != null && min2 != null) ? Math.round((min1 + min2) * 10) / 10 : null;
  var totalText = totalKm + " km routiers" + (totalMin != null ? " (~" + totalMin + " min)" : "");

  sheet.getRange(infoRow + 2, 1).setValue("Distance Tournée 1");
  sheet.getRange(infoRow + 2, 2).setValue(dist1Text);
  sheet.getRange(infoRow + 2, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 3, 1).setValue("Distance Tournée 2");
  sheet.getRange(infoRow + 3, 2).setValue(dist2Text);
  sheet.getRange(infoRow + 3, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 4, 1).setValue("Distance totale");
  sheet.getRange(infoRow + 4, 2).setValue(totalText);
  sheet.getRange(infoRow + 4, 1).setFontWeight("bold");
  sheet.getRange(infoRow + 4, 2).setFontWeight("bold");

  SpreadsheetApp.getActive().toast("Optimisation terminée !", "Résultat", 3);
}


// =========================
// BENCHMARK (historique cumulatif, jamais effacé)
// =========================
function _num(v) {
  return (v === null || v === undefined || v === "") ? "" : Number(v);
}


/**
 * Crée la feuille Benchmark si besoin, et complète l'en-tête des feuilles
 * antérieures au lot D-3 en n'écrivant QUE les colonnes manquantes, à la fin.
 * Aucune colonne existante n'est déplacée, renommée ni effacée.
 */
function ensureBenchmarkSheet() {

  const ss = SpreadsheetApp.getActive();
  let sheet = ss.getSheetByName(BENCH_SHEET);

  // Une feuille Apps Script naît avec 26 colonnes ; l'en-tête en compte 29
  // depuis le lot D-3. Sans cet élargissement, getRange et appendRow lèveraient
  // « out of bounds ».
  const widen = function (sh) {
    const missing = BENCH_HEADERS.length - sh.getMaxColumns();
    if (missing > 0) sh.insertColumnsAfter(sh.getMaxColumns(), missing);
  };

  if (!sheet) {
    sheet = ss.insertSheet(BENCH_SHEET);
    widen(sheet);
    sheet.getRange(1, 1, 1, BENCH_HEADERS.length).setValues([BENCH_HEADERS]);
    sheet.getRange(1, 1, 1, BENCH_HEADERS.length)
      .setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");
    sheet.setFrozenRows(1);
    sheet.setColumnWidth(1, 140);   // Date
    sheet.setColumnWidth(2, 150);   // Stratégie exécutée
    sheet.setColumnWidth(3, 150);   // Stratégie demandée
    sheet.setColumnWidth(17, 240);  // optimization_path
    return sheet;
  }

  // Migration : la feuille existe avec les 18 colonnes historiques.
  widen(sheet);
  const existing = sheet.getRange(1, 1, 1, BENCH_HEADERS.length).getValues()[0];
  const missing = [];
  for (var i = 0; i < BENCH_HEADERS.length; i++) {
    if (String(existing[i] || "").trim() === "") missing.push(i);
  }
  if (missing.length) {
    const first = missing[0];
    const tail = BENCH_HEADERS.slice(first);
    sheet.getRange(1, first + 1, 1, tail.length).setValues([tail]);
    sheet.getRange(1, first + 1, 1, tail.length)
      .setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");
  }
  return sheet;
}


/**
 * Ajoute une ligne à la feuille "Benchmark". Ne nettoie jamais, ne réécrit
 * jamais les lignes précédentes : c'est l'historique de comparaison.
 *
 * Les métriques de swaps sont lues directement dans la réponse du backend :
 * une optimisation normale les enregistre donc sans dépendre d'un label.
 *
 * @param {Object=} extra  {d3_label, run_error}. Absent ou partiel -> les
 *                         cellules correspondantes restent vides.
 */
function appendBenchmark(result, params, points, extra) {

  const sheet = ensureBenchmarkSheet();
  extra = extra || {};
  result = result || {};

  const km1 = _num(result.tournee_1_km);
  const km2 = _num(result.tournee_2_km);
  const min1 = _num(result.tournee_1_min);
  const min2 = _num(result.tournee_2_min);

  const kmTotal  = (km1 !== "" && km2 !== "")   ? Math.round((km1 + km2) * 100) / 100 : "";
  const minTotal = (min1 !== "" && min2 !== "") ? Math.round((min1 + min2) * 10) / 10 : "";

  const calls = result.api_calls || {};
  const sizes = result.partition_sizes || [];

  // Diagnostic de la stratégie expérimentale. Absent des quatre stratégies de
  // production : hyb vaut alors {} et toutes ses colonnes restent vides.
  const hyb = _hybridDiag(result);
  const stages = hyb.hybrid_stages || [];

  // Les champs D-3 sont absents des réponses d'un backend antérieur :
  // _cell() rend "" plutôt que de lever, les runs normaux restent valides.
  const row = [
    new Date(),
    // strategy_used = ce qui a RÉELLEMENT tourné. Un écart avec la colonne
    // suivante signale un repli et invalide la ligne comme point de comparaison.
    result.strategy_used || "",
    result.strategy_requested || params.strategy,
    points.length,
    result.points_signature || "",
    params.num_vehicles,
    km1, km2, kmTotal,
    min1, min2, minTotal,
    result.elapsed_ms != null ? Math.round(result.elapsed_ms / 100) / 10 : "",
    calls.total  != null ? calls.total  : "",
    calls.vroom  != null ? calls.vroom  : "",
    calls.matrix != null ? calls.matrix : "",
    result.optimization_path || "",
    sizes.join(" / "),

    // --- colonnes D-3 ---
    extra.d3_label || "",
    _cell(result.max_swap_candidates),
    _cell(result.swap_max_consecutive_fails),
    _cell(result.swap_candidates_tested),
    _cell(result.swaps_accepted),
    _cell(result.swap_resequence_cache_hits),
    _cell(result.swap_resequence_cache_misses),
    _cell(result.swap_vroom_calls_saved),
    _cell(result.swap_stop_reason),
    _cell(result.ortools_solution_limit),
    extra.run_error || "",

    // --- certificat territorial ---
    _cell(result.territorial_partition),
    _cell(result.territorial_method),
    _cell(result.territorial_membership_locked),
    _cell(result.territorial_side_violations),
    _cell(result.territorial_separator_angle_deg),
    _cell(result.territorial_separator_margin_m),
    _cell(result.territorial_overlap_status),
    _cell(result.territorial_candidates_unique),
    _cell(result.territorial_candidates_scored),
    _cell(result.territorial_fallback_used),
    _cell(result.territorial_error),

    // --- certificat de connexite ---
    _cell(result.connected_partition),
    _cell(result.connected_method),
    _cell(result.connected_membership_locked),
    _cellList(result.connected_target_sizes),
    _cell(result.connected_components_t1),
    _cell(result.connected_components_t2),
    _cellList(result.connected_component_sizes_t1),
    _cellList(result.connected_component_sizes_t2),
    _cell(result.connected_candidates_generated),
    _cell(result.connected_candidates_valid),
    _cell(result.connected_candidates_scored),
    _cell(result.connected_candidates_ortools),
    _cell(result.connected_candidates_vroom),
    _cell(result.connected_cut_edges),
    _cell(result.connected_cut_length_m),
    _cell(result.connected_cross_neighbors),
    _cell(result.connected_enclave_points),
    _cell(result.connected_selected_seed),
    _cell(result.connected_fallback_used),
    _cell(result.connected_error),
    _cell(result.connected_vroom_calls),
    _cell(result.selected_sequencer),
    _cell(result.final_selection_reason),
    _cell(result.ortools_total_duration_s),
    _cell(result.ortools_total_distance_m),
    _cell(result.vroom_total_duration_s),
    _cell(result.vroom_total_distance_m),
    _cell(result.partition_solver),

    // --- recentrage ORS-first ---
    _cell(result.connected_ors_reference_available),
    _cell(result.connected_ors_reference_duration_s),
    _cell(result.connected_ors_reference_distance_m),
    _cellList(result.connected_ors_reference_sizes),
    _cell(result.connected_ors_reference_components_t1),
    _cell(result.connected_ors_reference_components_t2),
    _cell(result.connected_ors_reference_time_limit_hit),
    _cell(result.connected_ors_reference_fallback_used),
    _cell(result.connected_ors_reference_solver_status),
    _cell(result.connected_ors_reference_solve_ms),
    _cell(result.connectivity_penalty_duration_s),
    _cell(result.connectivity_penalty_distance_m),
    _cell(result.connectivity_penalty_reliable),
    _cell(result.connectivity_penalty_note),
    _cell(result.connected_stage_timings_text),
    _cell(result.connected_stage_budget_exhausted),
    _cell(result.connected_generation_expired_after),
    _cell(result.connected_ors_repair_candidates_raw),
    _cell(result.connected_ors_repair_candidates_unique),
    _cell(result.connected_ors_repair_reached_ortools),
    _cell(result.connected_ors_repair_best_proxy_rank),
    _cell(result.connected_ors_repair_is_winner),
    _cell(result.connected_prescore_refined),
    _cell(result.connected_prescore_budget_exhausted),
    _cell(result.connected_winner_proxy_rank_rough),
    _cell(result.connected_winner_proxy_rank_refined),
    _cell(result.connected_winner_ortools_rank),
    _cell(result.connected_matrix_hash),
    _cell(result.connected_per_source_text),
    _cell(result.post_optimization_kept),
    _cell(result.post_optimization_note),

    // --- stratégie expérimentale VROOM local + ALNS territoriale ---
    // hyb est {} pour les quatre stratégies de production : toutes ces
    // cellules restent alors vides, aucune ligne existante ne change de forme.
    _cell(hyb.hybrid_error),
    _cell(hyb.local_vroom_enabled),
    _cell(hyb.local_vroom_version),
    _cell(hyb.common_rescore_duration_s),
    _cell(hyb.common_rescore_distance_m),
    _cell(hyb.common_rescore_matrix_hash),
    _cell(hyb.joint_direct_valid),
    _cell(hyb.joint_direct_duration_s),
    _cellList(hyb.joint_direct_sizes),
    _cell(hyb.joint_nucleus_attempted),
    _cell(hyb.joint_nucleus_valid),
    _cell(hyb.joint_nucleus_best_duration_s),
    _cell(hyb.route_first_unique),
    _cell(hyb.route_first_best_duration_s),
    _cell(hyb.joint_alns_iterations),
    _cell(hyb.joint_alns_accepted),
    _cell(hyb.joint_alns_seed),
    _cell(hyb.joint_alns_best_duration_s),
    _cell(hyb.joint_finalists),
    _cell(hyb.joint_finalists_local_vroom_solved),
    _cell(hyb.joint_finalists_reused),
    _cell(hyb.joint_solutions_considered),
    _cell(hyb.joint_selected_source),
    _cell(hyb.joint_selected_duration_s),
    _cell(hyb.joint_selected_distance_m),
    _cellList(hyb.joint_selected_sizes),
    _cellList(hyb.joint_selected_components),
    _cell(hyb.joint_selected_enclaves),
    _cell(hyb.joint_territorial_level),
    _cell(hyb.joint_territorial_max_enclaves),
    _cell(hyb.joint_territorial_admissible),
    _cell(hyb.joint_territorial_fallback_used),
    _cell(hyb.joint_territorial_fallback_reason),
    _cellList(hyb.joint_territorial_thresholds),
    _cellCounts(hyb.joint_territorial_level_counts),
    // Le plafond configuré, à lire en regard de local_vroom_attempted :
    // "4 lancées sur 4 autorisées" ne se voit pas sans lui.
    _cell(hyb.local_vroom_max_solves),
    _cell(hyb.local_vroom_attempted),
    _cell(hyb.local_vroom_succeeded),
    _cell(hyb.local_vroom_failed),
    _cell(hyb.local_vroom_timed_out),
    _cell(hyb.local_vroom_reused),
    _cell(hyb.local_vroom_skipped_for_time),
    _cell(hyb.local_vroom_elapsed_ms),
    _cell(hyb.local_vroom_stop_reason),
    _cell(hyb.local_vroom_last_error),
    _cell(_stageMs(stages, "matrix")),
    _cell(_stageMs(stages, "joint_direct")),
    _cell(_stageMs(stages, "route_first")),
    _cell(_stageMs(stages, "joint_nucleus")),
    _cell(_stageMs(stages, "joint_alns")),
    _cell(_stageMs(stages, "alns_refine")),
    _cell(_stageMs(stages, "joint_finalists")),
    // Texte de repli : si le backend ajoute un bloc, son temps apparaît ici
    // au lieu d'être perdu en silence.
    _stagesText(stages, "elapsed_ms"),
    _stagesText(stages, "stop_reason"),
    _cell(hyb.total_elapsed_ms),
    _cell(hyb.soft_limit_reached)
  ];

  sheet.appendRow(row);
}


/**
 * Bloc de diagnostic de la stratégie expérimentale.
 *
 * Il voyage dans result.ors_matrix.hybrid, là où la stratégie connexe passe
 * par result.ors_matrix.connected. Rend {} pour les quatre stratégies de
 * production, qui ne produisent pas ce bloc : les colonnes restent vides
 * plutôt que de lever.
 */
function _hybridDiag(result) {
  const meta = (result && result.ors_matrix) || {};
  return meta.hybrid || {};
}


/** Temps consommé par un bloc, "" si le bloc n'a pas tourné. */
function _stageMs(stages, name) {
  for (var i = 0; i < stages.length; i++) {
    if (stages[i] && stages[i].stage === name) return stages[i].elapsed_ms;
  }
  return "";
}


/** "bloc=valeur;bloc=valeur" sur tous les blocs réellement renvoyés. */
function _stagesText(stages, field) {
  const parts = [];
  for (var i = 0; i < stages.length; i++) {
    const stage = stages[i];
    if (!stage || !stage.stage) continue;
    parts.push(stage.stage + "=" + _cell(stage[field]));
  }
  return parts.join(";");
}


/** Sérialise un dictionnaire {clé: nombre} en "clé=valeur;clé=valeur". */
function _cellCounts(v) {
  if (v === null || v === undefined) return "";
  const keys = Object.keys(v).sort();
  const parts = [];
  for (var i = 0; i < keys.length; i++) {
    parts.push(keys[i] + "=" + v[keys[i]]);
  }
  return parts.join(";");
}


/** Rend "" pour null/undefined, la valeur sinon. Ne lève jamais. */
function _cell(v) {
  return (v === null || v === undefined) ? "" : v;
}


/** Sérialise une liste en texte lisible dans une cellule. "" si absente. */
function _cellList(v) {
  if (v === null || v === undefined) return "";
  return Array.isArray(v) ? v.join(" / ") : String(v);
}


// =========================
// CARTE DES DEUX TOURNÉES
// =========================
// Nom EXACT du fichier HTML à créer dans l'éditeur Apps Script
// (Fichier > Nouveau > Fichier HTML), sans l'extension.
const MAP_HTML_FILE  = "CarteTournees";

// Le payload d'un run de 62 points pèse ~8 Ko. PropertiesService plafonne à
// 9 Ko par valeur : trop juste dès que les adresses s'allongent. Une cellule
// de feuille accepte 50 000 caractères, avec en prime la persistance entre
// sessions. D'où cette feuille masquée à une seule cellule.
const MAP_DATA_SHEET = "_CarteData";

const MAP_COLORS = ["#1f5fa9", "#d35400"];   // Tournée 1 bleu, Tournée 2 orange foncé


function _mapDataSheet_(createIfMissing) {
  const ss = SpreadsheetApp.getActive();
  let sh = ss.getSheetByName(MAP_DATA_SHEET);
  if (!sh && createIfMissing) {
    sh = ss.insertSheet(MAP_DATA_SHEET);
    sh.getRange(1, 2).setValue(
      "Stockage technique de la dernière carte. Ne pas modifier.");
    sh.hideSheet();
  }
  return sh;
}


function _saveCartePayload_(payload) {
  _mapDataSheet_(true).getRange(1, 1).setValue(JSON.stringify(payload));
}


/**
 * Appelée depuis le HTML par google.script.run. Doit rester au niveau global.
 * Retourne le JSON du dernier run, ou null s'il n'y en a pas — le HTML affiche
 * alors « Aucune tournée récente disponible ».
 */
function getCarteTourneesPayload() {
  const sh = _mapDataSheet_(false);
  if (!sh) return null;
  const v = sh.getRange(1, 1).getValue();
  return v ? String(v) : null;
}


/**
 * Construit le payload de la carte à partir du run qui vient d'aboutir.
 *
 * result.tournee_N contient les identifiants DANS L'ORDRE DE PASSAGE, dépôt
 * inclus aux deux extrémités : le backend renvoie [start] + points + [end].
 * D'où role = "start" au premier rang, "end" au dernier, "collection" entre.
 *
 * Les latitudes, longitudes et libellés viennent du snapshot du Sheet passé en
 * argument — aucun appel réseau, aucun recalcul, aucune réoptimisation.
 */
function buildCartePayload(result, params, points) {

  const latMap = {}, lonMap = {}, addrMap = {};
  for (let i = 0; i < points.length; i++) {
    const sid = String(points[i].id);
    latMap[sid]  = points[i].lat;
    lonMap[sid]  = points[i].lon;
    addrMap[sid] = points[i].address || "";
  }

  const specs = [
    { key: "tournee_1", label: "Tournée 1" },
    { key: "tournee_2", label: "Tournée 2" }
  ];

  const routes = [];
  const skipped = [];

  for (let s = 0; s < specs.length; s++) {
    const ids = result[specs[s].key] || [];
    const pts = [];
    let order = 0;

    for (let k = 0; k < ids.length; k++) {
      const sid = String(ids[k]);
      const lat = Number(latMap[sid]);
      const lon = Number(lonMap[sid]);

      let role = "collection";
      if (k === 0) role = "start";
      else if (k === ids.length - 1) role = "end";

      // Coordonnée absente ou illisible : le point est écarté avec son
      // identifiant, sans empêcher l'affichage du reste de la carte.
      if (!isFinite(lat) || !isFinite(lon) || latMap[sid] === "" || lonMap[sid] === "") {
        skipped.push(specs[s].label + " / " + sid);
        continue;
      }

      if (role === "collection") order++;

      pts.push({
        order: (role === "start") ? "D" : (role === "end") ? "A" : order,
        id: sid,
        label: addrMap[sid],
        lat: lat,
        lon: lon,
        role: role
      });
    }

    routes.push({
      label: specs[s].label,
      color: MAP_COLORS[s % MAP_COLORS.length],
      distance_km:  _cell(result[specs[s].key + "_km"]),
      duration_min: _cell(result[specs[s].key + "_min"]),
      points: pts
    });
  }

  return {
    generated_at: new Date().toISOString(),
    strategy: result.strategy_used || result.strategy_requested || params.strategy || "",
    points_signature: result.points_signature || "",
    skipped: skipped,
    routes: routes,

    // Certificat territorial, affiché tel quel par la carte. Absent des
    // stratégies qui ne le produisent pas : la carte masque alors le bloc.
    territorial: {
      partition: result.territorial_partition,
      method: result.territorial_method,
      locked: result.territorial_membership_locked,
      violations: result.territorial_side_violations,
      angle_deg: result.territorial_separator_angle_deg,
      margin_m: result.territorial_separator_margin_m,
      status: result.territorial_overlap_status,
      error: result.territorial_error
    },

    // Certificat de connexite, affiché par la carte pour le mode connexe.
    connected: {
      partition: result.connected_partition,
      method: result.connected_method,
      locked: result.connected_membership_locked,
      sizes: result.connected_target_sizes,
      components_t1: result.connected_components_t1,
      components_t2: result.connected_components_t2,
      component_sizes_t1: result.connected_component_sizes_t1,
      component_sizes_t2: result.connected_component_sizes_t2,
      cut_edges: result.connected_cut_edges,
      cut_length_m: result.connected_cut_length_m,
      enclave_points: result.connected_enclave_points,
      sequencer: result.selected_sequencer,
      error: result.connected_error
    }
  };
}


function _ouvrirDialogueCarte_() {
  const out = HtmlService.createHtmlOutputFromFile(MAP_HTML_FILE)
    .setWidth(1200)
    .setHeight(800);
  SpreadsheetApp.getUi().showModalDialog(out, "Carte des deux tournées");
}


/** Construit, enregistre puis affiche la carte du run courant. */
function afficherCarteTournees(result, params, points) {
  _saveCartePayload_(buildCartePayload(result, params, points));
  _ouvrirDialogueCarte_();
}


/**
 * Fabrique une carte AUTONOME : le gabarit HTML avec les données du run
 * injectées dedans. Le fichier obtenu s'ouvre n'importe où — poste local,
 * pièce jointe, Drive — sans Apps Script et sans accès au classeur.
 *
 * L'amorçage du gabarit teste window.TOURNEES_PAYLOAD en premier : il rend
 * donc directement la carte et n'appelle jamais google.script.run.
 */
function _buildStandaloneCarteHtml_(payloadJson) {

  const tpl = HtmlService.createHtmlOutputFromFile(MAP_HTML_FILE).getContent();

  if (tpl.indexOf("</head>") === -1) {
    throw new Error("Gabarit " + MAP_HTML_FILE + " inattendu : balise </head> absente.");
  }

  // Les libellés viennent du Sheet et peuvent contenir n'importe quoi.
  // Échapper "<" neutralise </script> et <!-- , qui sortiraient du bloc.
  // U+2028 et U+2029 sont des terminateurs de ligne en JavaScript. Ils
  // doivent donc etre ecrits en forme echappee dans la regex : places tels
  // quels, ils cassent le litteral et le fichier ne se parse plus. Et jamais
  // l'espace ordinaire U+0020, qui n'a rien a neutraliser ici.
  const safe = payloadJson
    .replace(/</g, "\\u003c")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");

  return tpl.replace("</head>",
    "<script>window.TOURNEES_PAYLOAD = " + safe + ";</script>\n</head>");
}


/**
 * Entrée de menu : exporte la dernière carte en un fichier HTML autonome
 * déposé sur Drive, puis affiche son lien.
 *
 * Le fichier est créé PRIVÉ. Le partage reste une décision explicite, à
 * prendre dans Drive : ces données sont opérationnelles.
 */
function exporterCartePartageable() {

  const json = getCarteTourneesPayload();
  if (!json) {
    SpreadsheetApp.getActive().toast(
      "Aucune carte enregistrée. Lancez une optimisation.", "Export", 5);
    return;
  }

  const html = _buildStandaloneCarteHtml_(json);
  const stamp = Utilities.formatDate(
    new Date(), Session.getScriptTimeZone(), "yyyy-MM-dd_HH-mm");
  const name = "carte_tournees_" + stamp + ".html";

  const file = DriveApp.createFile(name, html, MimeType.HTML);

  const esc = function (s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  };

  const body =
      '<div style="font-family:Arial,sans-serif;font-size:13px;line-height:1.7;padding:14px">'
    + "<b>Carte exportée.</b><br>"
    + "Fichier : <code>" + esc(name) + "</code>"
    + " &mdash; " + Math.round(html.length / 1024) + " Ko<br><br>"
    + '<a href="' + esc(file.getUrl()) + '" target="_blank">Ouvrir dans Drive</a>'
    + " &nbsp;|&nbsp; "
    + '<a href="https://drive.google.com/uc?export=download&id='
    + esc(file.getId()) + '" target="_blank">Télécharger le fichier</a>'
    + "<br><br>"
    + "<b>Pour partager</b> : dans Drive, clic droit sur le fichier &rsaquo; Partager.<br>"
    + "Il est créé <b>privé</b> : rien n'est publié sans votre décision.<br><br>"
    + "<small>Le fichier contient les données du run. Il charge Leaflet et les "
    + "tuiles OpenStreetMap depuis Internet : une connexion reste nécessaire "
    + "pour l'afficher.</small></div>";

  SpreadsheetApp.getUi().showModalDialog(
    HtmlService.createHtmlOutput(body).setWidth(560).setHeight(300),
    "Carte partageable");
}


/** Entrée de menu : rouvre la dernière carte sans relancer d'optimisation. */
function afficherDerniereCarte() {
  if (!getCarteTourneesPayload()) {
    SpreadsheetApp.getActive().toast(
      "Aucune carte enregistrée. Lancez une optimisation.", "Carte", 5);
    return;
  }
  _ouvrirDialogueCarte_();
}


// =========================
// LANCER L'OPTIMISATION
// =========================
/**
 * @param {string=} strategyOverride  Forcé par les entrées de menu.
 *                                    Absent -> stratégie lue dans Paramètres!B6.
 * Une seule stratégie par exécution : enchaîner les trois dépasserait
 * la limite de 6 minutes d'Apps Script.
 */
function runOptimisation(strategyOverride) {

  ensureStrategyCell();

  const params = getParams();
  if (typeof strategyOverride === "string" && strategyOverride) {
    params.strategy = strategyOverride;
  }

  const points = getPoints();

  if (points.length === 0) {
    SpreadsheetApp.getActive().toast("Aucun point sélectionné !", "Erreur", 3);
    return;
  }

  SpreadsheetApp.getActive().toast(
    "Optimisation en cours... (" + points.length + " points, stratégie " + params.strategy + ")",
    "Info", 10
  );

  const result = callAPI(points, params);

  if (result.error) {
    SpreadsheetApp.getActive().toast("Erreur API : " + result.error, "Erreur", 5);
    return;
  }

  writeResult(result, params, points);

  // Une réponse 200 peut masquer une dégradation backend : VROOM indisponible
  // => vroom_ok faux => swaps jamais exécutés. Sans cette remontée, la ligne
  // Benchmark paraît normale alors qu'elle ne mesure pas la même chose.
  var degraded = "";
  if (result.vroom_used === false) {
    degraded = "VROOM indisponible (" + (result.vroom_error || "raison inconnue")
             + ") : swaps non exécutés";
  } else if (result.swap_stop_reason === "vroom_error") {
    degraded = "swaps non exécutés (vroom_error)";
  }

  appendBenchmark(result, params, points, { run_error: degraded });

  var vroomInfo = result.vroom_used ? "Vroom direct" : "K-Means + Vroom (fallback)";
  var stratLabel = result.strategy_used || params.strategy;
  SpreadsheetApp.getActive().toast(
    "Terminé ! Stratégie : " + stratLabel + " | " + vroomInfo,
    "Succès", 10
  );

  // Carte en DERNIER : un dialogue modal bloque le script jusqu'à sa fermeture,
  // les résultats doivent donc déjà être écrits. On n'arrive ici qu'après un run
  // réussi : callAPI lève sur une erreur HTTP et result.error sort plus haut,
  // donc aucune carte n'est ouverte après un échec.
  try {
    afficherCarteTournees(result, params, points);
  } catch (e) {
    SpreadsheetApp.getActive().toast(
      "Carte non affichée : " + (e && e.message ? e.message : e), "Carte", 8);
  }
}


// Wrappers : Apps Script n'autorise pas d'argument depuis une entrée de menu.
function runKmeans()           { runOptimisation("kmeans"); }
function runOrtoolsHaversine() { runOptimisation("ortools_haversine"); }
function runOrtoolsOrsMatrix() { runOptimisation("ortools_ors_matrix"); }
function runOrtoolsOrsMatrixConnected() { runOptimisation("ortools_ors_matrix_connected"); }

// Stratégie expérimentale. L'identifiant vient de la constante, jamais d'une
// chaîne recopiée : le libellé de menu et le nom envoyé à l'API ne peuvent
// pas diverger.
function runHybridLocalVroomTerritorial() { runOptimisation(EXP_STRATEGY); }



// =========================
// MENU PERSONNALISÉ
// =========================
function onOpen() {
  const ui = SpreadsheetApp.getUi();
  ui.createMenu("Tournées")
    .addItem("Optimisation", "runOptimisation")
    .addSubMenu(
      ui.createMenu("Optimiser avec")
        .addItem("K-Means (baseline)", "runKmeans")
        .addItem("OR-Tools Haversine", "runOrtoolsHaversine")
        .addItem("OR-Tools ORS Matrix", "runOrtoolsOrsMatrix")
        .addItem("OR-Tools ORS Matrix — territoires connexes",
                 "runOrtoolsOrsMatrixConnected")
        .addSeparator()
        .addItem(EXP_STRATEGY_LABEL, "runHybridLocalVroomTerritorial")
    )
    .addSeparator()
    .addItem("Afficher la dernière carte", "afficherDerniereCarte")
    .addItem("Exporter la carte partageable", "exporterCartePartageable")
    .addSeparator()
    .addItem("Effacer tournées", "clearResults")
    .addItem("Réinitialiser la sélection", "resetSelection")
    .addToUi();
}


// =========================
// EFFACER TOURNÉES
// =========================
// N'efface que "Résultats". La feuille "Benchmark" est volontairement épargnée :
// c'est l'historique de comparaison des stratégies.
function clearResults() {
  const sheet = SpreadsheetApp.getActive().getSheetByName("Résultats");
  if (sheet) {
    sheet.clear();
    SpreadsheetApp.getActive().toast("Tournées effacées !", "Info", 3);
  }
}


// =========================
// RÉINITIALISER SÉLECTION
// =========================
function resetSelection() {
  const sheet = SpreadsheetApp.getActive().getSheetByName("Horodateurs");
  if (!sheet) return;

  const lastRow = sheet.getLastRow();
  if (lastRow < 2) return;

  sheet.getRange(2, 5, lastRow - 1, 1).uncheck();
  SpreadsheetApp.getActive().toast("Sélection réinitialisée !", "Info", 3);
}