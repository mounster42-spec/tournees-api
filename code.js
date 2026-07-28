// =========================
// CONSTANTES
// =========================
const API_BASE = "https://tournees-api.onrender.com";

const STRATEGIES = ["kmeans", "ortools_haversine", "ortools_ors_matrix"];
const DEFAULT_STRATEGY = "kmeans";

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
  "ortools_ors_matrix": "OR-Tools ORS Matrix (affectation) + Vroom (séquencement)"
};

// Repli si partition_engine est absent (backend antérieur au lot 3).
const STRATEGY_LABELS = {
  "kmeans":             "K-Means (affectation) + Vroom (séquencement)",
  "ortools_haversine":  "OR-Tools Haversine (affectation) + Vroom (séquencement)",
  "ortools_ors_matrix": "OR-Tools ORS Matrix (affectation) + Vroom (séquencement)"
};

const BENCH_SHEET = "Benchmark";
const BENCH_HEADERS = [
  "Date", "Stratégie exécutée", "Stratégie demandée", "Nb pts", "Signature jeu", "Nb véh",
  "Km T1", "Km T2", "Km total",
  "Min T1", "Min T2", "Min total",
  "Temps calcul (s)", "Appels API", "Vroom", "Matrix",
  "optimization_path", "Répartition"
];


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
 * Ajoute une ligne à la feuille "Benchmark". Ne nettoie jamais, ne réécrit
 * jamais les lignes précédentes : c'est l'historique de comparaison.
 */
function appendBenchmark(result, params, points) {

  const ss = SpreadsheetApp.getActive();
  let sheet = ss.getSheetByName(BENCH_SHEET);

  if (!sheet) {
    sheet = ss.insertSheet(BENCH_SHEET);
    sheet.getRange(1, 1, 1, BENCH_HEADERS.length).setValues([BENCH_HEADERS]);
    sheet.getRange(1, 1, 1, BENCH_HEADERS.length)
      .setBackground("#434343").setFontColor("#ffffff").setFontWeight("bold");
    sheet.setFrozenRows(1);
    sheet.setColumnWidth(1, 140);   // Date
    sheet.setColumnWidth(2, 150);   // Stratégie exécutée
    sheet.setColumnWidth(3, 150);   // Stratégie demandée
    sheet.setColumnWidth(17, 240);  // optimization_path
  }

  const km1 = _num(result.tournee_1_km);
  const km2 = _num(result.tournee_2_km);
  const min1 = _num(result.tournee_1_min);
  const min2 = _num(result.tournee_2_min);

  const kmTotal  = (km1 !== "" && km2 !== "")   ? Math.round((km1 + km2) * 100) / 100 : "";
  const minTotal = (min1 !== "" && min2 !== "") ? Math.round((min1 + min2) * 10) / 10 : "";

  const calls = result.api_calls || {};
  const sizes = result.partition_sizes || [];

  sheet.appendRow([
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
    sizes.join(" / ")
  ]);
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
  appendBenchmark(result, params, points);

  var vroomInfo = result.vroom_used ? "Vroom direct" : "K-Means + Vroom (fallback)";
  var stratLabel = result.strategy_used || params.strategy;
  SpreadsheetApp.getActive().toast(
    "Terminé ! Stratégie : " + stratLabel + " | " + vroomInfo,
    "Succès", 10
  );
}


// Wrappers : Apps Script n'autorise pas d'argument depuis une entrée de menu.
function runKmeans()           { runOptimisation("kmeans"); }
function runOrtoolsHaversine() { runOptimisation("ortools_haversine"); }
function runOrtoolsOrsMatrix() { runOptimisation("ortools_ors_matrix"); }


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
    )
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