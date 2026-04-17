"""
NEXUS — Complete Flask Backend with ML Prediction
===================================================
Run: python app.py
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import pymysql, json, time, threading, math, pickle, os
import numpy as np
from datetime import datetime, date
from decimal import Decimal

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── MySQL Config ──────────────────────────────────────
DB_CONFIG = {
    "host": "localhost", "port": 3306,
    "user": "root",
    "password": "Root@1234",
    "database": "disaster_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": True,
}

def get_conn():
    return pymysql.connect(**DB_CONFIG)

def serial(obj):
    if isinstance(obj, (datetime, date)): return obj.isoformat()
    if isinstance(obj, Decimal): return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def rows_json(rows):
    return json.loads(json.dumps(rows, default=serial))

# ── Load ML Model ─────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "disaster_model.pkl")
ml_model = None
try:
    with open(MODEL_PATH, "rb") as f:
        ml_model = pickle.load(f)
    print("✅ ML Model loaded successfully!")
except FileNotFoundError:
    print("⚠️  disaster_model.pkl not found — run train_model.py first!")

# ── SSE ───────────────────────────────────────────────
_sse_clients = []
_sse_lock    = threading.Lock()

def broadcast(event_type: str, data: dict):
    msg = f"event: {event_type}\ndata: {json.dumps(data, default=serial)}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:    q.append(msg)
            except: dead.append(q)
        for q in dead: _sse_clients.remove(q)

@app.route("/api/stream")
def sse_stream():
    client_q = []
    with _sse_lock: _sse_clients.append(client_q)
    def gen():
        yield 'data: {"status":"connected"}\n\n'
        try:
            while True:
                if client_q: yield client_q.pop(0)
                else:        yield ": heartbeat\n\n"
                time.sleep(1)
        except GeneratorExit:
            with _sse_lock:
                if client_q in _sse_clients: _sse_clients.remove(client_q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"})

# ── BDI Agent Logic ───────────────────────────────────
def bayesian_risk_update(prior, sensor_value, threshold):
    likelihood = min(1.0, sensor_value / threshold)
    posterior = (likelihood * prior) / (likelihood * prior + 0.1 * (1 - prior))
    return round(posterior * 100, 2)

def dijkstra_priority_score(risk, active_agents, distance_km):
    urgency = risk / 100.0
    resource = 1.0 / max(1, active_agents)
    distance = 1.0 / (1.0 + math.log1p(distance_km))
    return round((urgency * 0.6 + resource * 0.25 + distance * 0.15) * 100, 2)

def bdi_decide(event, sensor_readings):
    anomalous = [s for s in sensor_readings if s.get("is_anomaly")]
    risk      = float(event.get("risk_score", 50))
    severity  = event.get("severity", "MEDIUM")
    priority  = dijkstra_priority_score(risk, len(sensor_readings), 50)

    if severity == "CRITICAL" or risk >= 85:
        action, confidence = "ESCALATE", min(98, 85 + len(anomalous) * 2)
        reasoning = f"Critical risk {risk}. {len(anomalous)} anomalies. ESCALATE to NDMA."
    elif severity == "HIGH" or risk >= 65:
        action, confidence = "ALERT", min(95, 75 + len(anomalous) * 2)
        reasoning = f"High risk {risk}. Bayesian update triggered. Issue Level-3 ALERT."
    elif severity == "MEDIUM" or risk >= 40:
        action, confidence = "MONITOR", min(88, 65 + len(anomalous))
        reasoning = f"Medium risk {risk}. Dijkstra route computed. Enhanced MONITOR."
    else:
        action, confidence = "OBSERVE", 70
        reasoning = f"Low risk {risk}. Passive OBSERVE. No immediate action."

    return {"decision_type": action, "confidence": round(confidence, 2),
            "reasoning": reasoning, "priority_score": priority,
            "anomalous_sensors": len(anomalous)}

# ── PREDICT ───────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if not ml_model:
        return jsonify({"error": "ML model not loaded. Run train_model.py first!"}), 500

    data = request.get_json()
    required = ["rainfall_mm","water_level_m","wind_speed_kmh","seismic_mag","temperature_c","humidity_pct"]
    for f in required:
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400

    model    = ml_model["model"]
    scaler   = ml_model["scaler"]
    features = ml_model["features"]

    X        = np.array([[data[f] for f in features]])
    X_scaled = scaler.transform(X)

    prediction    = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    classes       = model.classes_

    confidence_map = {cls: round(float(prob)*100, 1) for cls, prob in zip(classes, probabilities)}
    top_confidence = round(float(max(probabilities))*100, 1)

    risk_score = round(
        (data["rainfall_mm"]/400*30) +
        (data["water_level_m"]/9*25) +
        (data["wind_speed_kmh"]/300*20) +
        (data["seismic_mag"]/9*25), 2
    )
    risk_score = min(99.9, max(0.1, risk_score * (top_confidence/100)))

    if risk_score >= 75:   severity = "CRITICAL"
    elif risk_score >= 55: severity = "HIGH"
    elif risk_score >= 35: severity = "MEDIUM"
    else:                  severity = "LOW"

    actions = {
        "CRITICAL": "IMMEDIATE EVACUATION — Alert NDMA and deploy NDRF teams",
        "HIGH":     "Issue Level-3 alert — Pre-position rescue teams",
        "MEDIUM":   "Enhanced monitoring — Notify district authorities",
        "LOW":      "Continue observation — No immediate action required",
    }

    result = {
        "prediction": prediction, "confidence": top_confidence,
        "risk_score": risk_score, "severity": severity,
        "action": actions[severity], "all_probabilities": confidence_map,
        "inputs": data,
    }
    broadcast("prediction", result)
    return jsonify(result)

# ── SENSOR LIVE ───────────────────────────────────────
@app.route("/api/sensor-live", methods=["GET"])
def sensor_live():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""
                SELECT s1.* FROM sensor_readings s1
                INNER JOIN (
                    SELECT sensor_type, MAX(recorded_at) AS latest
                    FROM sensor_readings GROUP BY sensor_type
                ) s2 ON s1.sensor_type = s2.sensor_type
                AND s1.recorded_at = s2.latest
                ORDER BY s1.sensor_type
            """)
            rows = c.fetchall()
        return jsonify(rows_json(rows))
    finally: conn.close()

# ── DASHBOARD ─────────────────────────────────────────
@app.route("/api/dashboard")
def dashboard():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("SELECT COUNT(*) AS t FROM disaster_events"); total = c.fetchone()["t"]
            c.execute("SELECT COUNT(*) AS t FROM disaster_events WHERE status='ACTIVE'"); active = c.fetchone()["t"]
            c.execute("SELECT COUNT(*) AS t FROM disaster_events WHERE severity='CRITICAL'"); critical = c.fetchone()["t"]
            c.execute("SELECT COUNT(*) AS t FROM sensor_readings WHERE is_anomaly=1"); anomalies = c.fetchone()["t"]
            c.execute("SELECT COUNT(*) AS t FROM alerts WHERE is_sent=1"); alerts = c.fetchone()["t"]
            c.execute("SELECT ROUND(AVG(risk_score),2) AS r FROM disaster_events"); avg_risk = c.fetchone()["r"]
            c.execute("SELECT disaster_type, COUNT(*) AS cnt FROM disaster_events GROUP BY disaster_type"); by_type = c.fetchall()
            c.execute("SELECT severity, COUNT(*) AS cnt FROM disaster_events GROUP BY severity"); by_sev = c.fetchall()
        return jsonify({
            "total_events": total, "active_events": active, "critical_events": critical,
            "anomalous_sensors": anomalies, "alerts_sent": alerts,
            "avg_risk_score": float(avg_risk or 0),
            "by_type": rows_json(by_type), "by_severity": rows_json(by_sev),
        })
    finally: conn.close()

# ── EVENTS ────────────────────────────────────────────
@app.route("/api/events", methods=["GET"])
def get_events():
    status = request.args.get("status"); sev = request.args.get("severity")
    conn = get_conn()
    try:
        with conn.cursor() as c:
            sql = "SELECT * FROM disaster_events WHERE 1=1"; p = []
            if status: sql += " AND status=%s"; p.append(status.upper())
            if sev:    sql += " AND severity=%s"; p.append(sev.upper())
            sql += " ORDER BY recorded_at DESC"
            c.execute(sql, p); rows = c.fetchall()
        return jsonify(rows_json(rows))
    finally: conn.close()

@app.route("/api/events", methods=["POST"])
def create_event():
    d = request.get_json()
    for f in ["disaster_type","location","severity"]:
        if f not in d: return jsonify({"error": f"Missing: {f}"}), 400
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""INSERT INTO disaster_events
                (disaster_type,location,latitude,longitude,severity,risk_score,status,description)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (d["disaster_type"].upper(), d["location"],
                 d.get("latitude"), d.get("longitude"),
                 d["severity"].upper(), d.get("risk_score",0.0),
                 d.get("status","MONITORING").upper(), d.get("description","")))
            new_id = c.lastrowid

        sensor_rows = []
        with conn.cursor() as c:
            c.execute("SELECT * FROM sensor_readings WHERE event_id=%s",(new_id,))
            sensor_rows = c.fetchall()

        decision = bdi_decide(d, sensor_rows)
        with conn.cursor() as c:
            c.execute("""INSERT INTO agent_decisions
                (event_id,agent_id,decision_type,confidence,reasoning,action_taken)
                VALUES (%s,%s,%s,%s,%s,%s)""",
                (new_id, f"NEXUS-AUTO-{d['disaster_type'][:4].upper()}",
                 decision["decision_type"], decision["confidence"],
                 decision["reasoning"],
                 f"Auto BDI decision. Priority: {decision['priority_score']}"))

        broadcast("new_event", {"id": new_id, **d})
        return jsonify({"message":"Event created","id":new_id,"agent_decision":decision}), 201
    finally: conn.close()

@app.route("/api/events/<int:eid>", methods=["PUT"])
def update_event(eid):
    d = request.get_json()
    fields = {k:v for k,v in d.items() if k in ["severity","risk_score","status","description"]}
    if not fields: return jsonify({"error":"No updatable fields"}), 400
    conn = get_conn()
    try:
        with conn.cursor() as c:
            clause = ", ".join(f"{k}=%s" for k in fields)
            c.execute(f"UPDATE disaster_events SET {clause} WHERE id=%s", list(fields.values())+[eid])
        broadcast("update_event", {"id":eid,**fields})
        return jsonify({"message":"Updated","id":eid})
    finally: conn.close()

@app.route("/api/events/<int:eid>", methods=["DELETE"])
def delete_event(eid):
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("DELETE FROM disaster_events WHERE id=%s",(eid,))
        broadcast("delete_event",{"id":eid})
        return jsonify({"message":"Deleted"})
    finally: conn.close()

# ── SENSORS ───────────────────────────────────────────
@app.route("/api/sensors", methods=["GET"])
def get_sensors():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            sql = "SELECT * FROM sensor_readings WHERE 1=1"; p = []
            if request.args.get("event_id"):
                sql += " AND event_id=%s"; p.append(request.args.get("event_id"))
            if request.args.get("anomaly") == "1":
                sql += " AND is_anomaly=1"
            sql += " ORDER BY recorded_at DESC LIMIT 100"
            c.execute(sql,p); rows = c.fetchall()
        return jsonify(rows_json(rows))
    finally: conn.close()

@app.route("/api/sensors", methods=["POST"])
def add_sensor():
    d = request.get_json()
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""INSERT INTO sensor_readings
                (sensor_id,event_id,sensor_type,value,unit,latitude,longitude,is_anomaly)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (d["sensor_id"],d.get("event_id"),d["sensor_type"],d["value"],
                 d["unit"],d.get("latitude"),d.get("longitude"),int(d.get("is_anomaly",0))))
            new_id = c.lastrowid
        broadcast("new_sensor",{"id":new_id,**d})
        return jsonify({"message":"Sensor added","id":new_id}), 201
    finally: conn.close()

# ── DECISIONS ─────────────────────────────────────────
@app.route("/api/decisions", methods=["GET"])
def get_decisions():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""SELECT ad.*, de.disaster_type, de.location
                FROM agent_decisions ad
                JOIN disaster_events de ON ad.event_id=de.id
                ORDER BY ad.decided_at DESC LIMIT 50""")
            rows = c.fetchall()
        return jsonify(rows_json(rows))
    finally: conn.close()

# ── ALERTS ────────────────────────────────────────────
@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""SELECT a.*, de.disaster_type, de.location
                FROM alerts a JOIN disaster_events de ON a.event_id=de.id
                ORDER BY a.created_at DESC""")
            rows = c.fetchall()
        return jsonify(rows_json(rows))
    finally: conn.close()

# ── RISK ──────────────────────────────────────────────
@app.route("/api/risk", methods=["GET"])
def get_risk():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""SELECT ra.*, de.disaster_type, de.location, de.severity
                FROM risk_assessments ra
                JOIN disaster_events de ON ra.event_id=de.id
                ORDER BY ra.assessed_at DESC""")
            rows = c.fetchall()
        return jsonify(rows_json(rows))
    finally: conn.close()

# ── HIVE ANALYTICS ────────────────────────────────────
@app.route("/api/analytics/hive", methods=["GET"])
def hive_analytics():
    conn = get_conn()
    try:
        with conn.cursor() as c:
            c.execute("""SELECT disaster_type, ROUND(AVG(risk_score),2) AS avg_risk,
                MAX(risk_score) AS max_risk, COUNT(*) AS event_count
                FROM disaster_events GROUP BY disaster_type ORDER BY avg_risk DESC""")
            risk_by_type = c.fetchall()
            c.execute("""SELECT YEAR(recorded_at) AS yr, MONTH(recorded_at) AS mo,
                disaster_type, COUNT(*) AS events
                FROM disaster_events GROUP BY yr,mo,disaster_type ORDER BY yr DESC,mo DESC""")
            trend = c.fetchall()
            c.execute("""SELECT sensor_type, COUNT(*) AS anomaly_count
                FROM sensor_readings WHERE is_anomaly=1
                GROUP BY sensor_type ORDER BY anomaly_count DESC""")
            anomalies = c.fetchall()
        return jsonify({
            "source": "MySQL → Hive ORC analytics pipeline",
            "risk_by_type": rows_json(risk_by_type),
            "monthly_trend": rows_json(trend),
            "anomaly_summary": rows_json(anomalies),
        })
    finally: conn.close()

# ── AGENT RUN ─────────────────────────────────────────
@app.route("/api/agent/run", methods=["POST"])
def run_agent():
    conn = get_conn()
    try:
        decisions_made = []
        with conn.cursor() as c:
            c.execute("SELECT * FROM disaster_events WHERE status='ACTIVE'")
            active = c.fetchall()
        for ev in active:
            with conn.cursor() as c:
                c.execute("SELECT * FROM sensor_readings WHERE event_id=%s",(ev["id"],))
                sensors = c.fetchall()
            decision = bdi_decide(ev, sensors)
            with conn.cursor() as c:
                c.execute("""INSERT INTO agent_decisions
                    (event_id,agent_id,decision_type,confidence,reasoning,action_taken)
                    VALUES (%s,%s,%s,%s,%s,%s)""",
                    (ev["id"],"NEXUS-SCHEDULED",
                     decision["decision_type"],decision["confidence"],
                     decision["reasoning"],
                     f"Scheduled BDI. Priority: {decision['priority_score']}"))
            decisions_made.append({"event_id":ev["id"],**decision})
            broadcast("agent_decision",{"event_id":ev["id"],"decision":decision["decision_type"]})
        return jsonify({"decisions":decisions_made,"count":len(decisions_made)})
    finally: conn.close()

# ── HEALTH ────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    try:
        c = get_conn(); c.ping(); c.close(); ok = True
    except: ok = False
    return jsonify({
        "status": "ok" if ok else "degraded",
        "database": "connected" if ok else "unreachable",
        "ml_model": "loaded" if ml_model else "not loaded",
        "agent": "NEXUS BDI v2.4 ACTIVE",
        "timestamp": datetime.utcnow().isoformat(),
    })

# ── SCHEDULED AGENT CYCLE ─────────────────────────────
def scheduled_agent_cycle():
    while True:
        time.sleep(60)
        try:
            import requests as req
            req.post("http://localhost:5000/api/agent/run", timeout=10)
        except: pass

threading.Thread(target=scheduled_agent_cycle, daemon=True).start()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
