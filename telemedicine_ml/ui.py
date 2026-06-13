import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model import predict_churn, predict_risk

# ── Theme ─────────────────────────────────────────────────────────────────────
BG       = "#0a0f1e"
BG2      = "#0d1526"
BG3      = "#111827"
BORDER   = "#1e2d45"
ACCENT   = "#38bdf8"
ACCENT2  = "#818cf8"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"
DANGER   = "#f43f5e"
SUCCESS  = "#22c55e"
WARNING  = "#f59e0b"
FONT     = "Segoe UI"


def styled_label(parent, text, size=10, color=TEXT, bold=False, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, font=(FONT, size, weight),
                    bg=parent["bg"] if hasattr(parent, "__getitem__") else BG2,
                    fg=color, **kw)


def styled_entry(parent, width=18):
    e = tk.Entry(parent, width=width, font=(FONT, 10),
                 bg=BG3, fg=TEXT, insertbackground=TEXT,
                 relief="flat", bd=0, highlightthickness=1,
                 highlightbackground=BORDER, highlightcolor=ACCENT)
    return e


def styled_combo(parent, values, width=16):
    style = ttk.Style()
    style.configure("Dark.TCombobox",
                    fieldbackground=BG3, background=BG3,
                    foreground=TEXT, selectbackground=BG3,
                    selectforeground=TEXT, borderwidth=0)
    cb = ttk.Combobox(parent, values=values, width=width,
                      font=(FONT, 10), style="Dark.TCombobox", state="readonly")
    cb.current(0)
    return cb


def card(parent, **kw):
    f = tk.Frame(parent, bg=BG2, highlightthickness=1,
                 highlightbackground=BORDER, **kw)
    return f


def section_title(parent, text, color=ACCENT):
    tk.Label(parent, text=text, font=(FONT, 9, "bold"),
             bg=BG2, fg=color).pack(anchor="w", pady=(0, 8))
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("MediAI — Telemedicine ML Platform")
root.geometry("1100x720")
root.configure(bg=BG)
root.resizable(True, True)

# ttk global style
style = ttk.Style()
style.theme_use("clam")
style.configure("TNotebook", background=BG, borderwidth=0)
style.configure("TNotebook.Tab", background=BG2, foreground=MUTED,
                font=(FONT, 10, "bold"), padding=[18, 8])
style.map("TNotebook.Tab",
          background=[("selected", BG3)],
          foreground=[("selected", ACCENT)])
style.configure("TFrame", background=BG)
style.configure("Vertical.TScrollbar", background=BG2, troughcolor=BG,
                borderwidth=0, arrowcolor=MUTED)

# ── Header ────────────────────────────────────────────────────────────────────
header = tk.Frame(root, bg="#0d1f3c", height=64)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(header, text="🏥", font=(FONT, 20), bg="#0d1f3c", fg=TEXT).pack(side="left", padx=(20, 6), pady=10)
tk.Label(header, text="MediAI", font=(FONT, 16, "bold"), bg="#0d1f3c", fg=ACCENT).pack(side="left")
tk.Label(header, text="  Telemedicine Intelligence Platform — India",
         font=(FONT, 10), bg="#0d1f3c", fg=MUTED).pack(side="left", pady=14)

# ── Notebook tabs ─────────────────────────────────────────────────────────────
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True, padx=0, pady=0)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
tab_home = tk.Frame(nb, bg=BG)
nb.add(tab_home, text="  🏠  Home  ")

canvas_h = tk.Canvas(tab_home, bg=BG, highlightthickness=0)
scrollbar_h = ttk.Scrollbar(tab_home, orient="vertical", command=canvas_h.yview)
scroll_frame_h = tk.Frame(canvas_h, bg=BG)
scroll_frame_h.bind("<Configure>", lambda e: canvas_h.configure(scrollregion=canvas_h.bbox("all")))
canvas_h.create_window((0, 0), window=scroll_frame_h, anchor="nw")
canvas_h.configure(yscrollcommand=scrollbar_h.set)
scrollbar_h.pack(side="right", fill="y")
canvas_h.pack(side="left", fill="both", expand=True)
canvas_h.bind("<MouseWheel>", lambda e: canvas_h.yview_scroll(int(-1*(e.delta/120)), "units"))

inner = scroll_frame_h

# Hero
hero = tk.Frame(inner, bg="#0d1f3c", padx=40, pady=30)
hero.pack(fill="x", padx=20, pady=(20, 14))
tk.Label(hero, text="🇮🇳  India Telemedicine Intelligence",
         font=(FONT, 9, "bold"), bg="#0d1f3c", fg=ACCENT).pack(anchor="w")
tk.Label(hero, text="AI-Powered Healthcare Decision Platform",
         font=(FONT, 20, "bold"), bg="#0d1f3c", fg=TEXT).pack(anchor="w", pady=(6, 4))
tk.Label(hero, text="Predict patient churn, assess clinical risk, and explore analytics\nacross India's post-pandemic telemedicine ecosystem.",
         font=(FONT, 10), bg="#0d1f3c", fg=MUTED, justify="left").pack(anchor="w")

# KPI row
try:
    import pandas as pd
    patients     = pd.read_csv("data/patients.csv")
    consultations= pd.read_csv("data/consultations.csv")
    kpi_data = [
        ("👥", f"{len(patients):,}", "Total Patients", ACCENT),
        ("📉", f"{patients['churned'].mean()*100:.1f}%", "Churn Rate", DANGER),
        ("⭐", f"{patients['satisfaction_score'].mean():.2f}/5", "Avg Satisfaction", SUCCESS),
        ("🚨", f"{consultations['high_risk'].sum():,}", "High Risk Cases", WARNING),
    ]
except Exception:
    kpi_data = [
        ("👥", "—", "Total Patients", ACCENT),
        ("📉", "—", "Churn Rate", DANGER),
        ("⭐", "—", "Avg Satisfaction", SUCCESS),
        ("🚨", "—", "High Risk Cases", WARNING),
    ]

kpi_row = tk.Frame(inner, bg=BG)
kpi_row.pack(fill="x", padx=20, pady=(0, 14))
for icon, val, lbl, color in kpi_data:
    c = card(kpi_row, padx=24, pady=16)
    c.pack(side="left", expand=True, fill="both", padx=6)
    tk.Label(c, text=icon, font=(FONT, 20), bg=BG2, fg=color).pack(anchor="w")
    tk.Label(c, text=val, font=(FONT, 18, "bold"), bg=BG2, fg=TEXT).pack(anchor="w")
    tk.Label(c, text=lbl, font=(FONT, 9), bg=BG2, fg=MUTED).pack(anchor="w")
    tk.Frame(c, bg=color, height=3).pack(fill="x", side="bottom")

# Feature cards
feat_row = tk.Frame(inner, bg=BG)
feat_row.pack(fill="x", padx=20, pady=(0, 20))
features = [
    ("📊", "Analytics", "Explore churn trends, platform distribution, and risk patterns.", ACCENT),
    ("🔮", "Churn Predictor", "Predict which patients are at risk of leaving the platform.", ACCENT2),
    ("⚕️", "Risk Classifier", "Assess clinical risk from symptoms and vitals in real-time.", WARNING),
]
for icon, title, desc, color in features:
    fc = card(feat_row, padx=22, pady=20)
    fc.pack(side="left", expand=True, fill="both", padx=6)
    tk.Frame(fc, bg=color, height=3).pack(fill="x")
    tk.Label(fc, text=icon, font=(FONT, 22), bg=BG2).pack(pady=(12, 4))
    tk.Label(fc, text=title, font=(FONT, 11, "bold"), bg=BG2, fg=TEXT).pack()
    tk.Label(fc, text=desc, font=(FONT, 9), bg=BG2, fg=MUTED,
             wraplength=220, justify="center").pack(pady=(4, 10))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
tab_churn = tk.Frame(nb, bg=BG)
nb.add(tab_churn, text="  🔮  Churn Predictor  ")

churn_main = tk.Frame(tab_churn, bg=BG)
churn_main.pack(fill="both", expand=True, padx=24, pady=20)

# Left form
form_c = card(churn_main, padx=22, pady=18)
form_c.pack(side="left", fill="both", expand=True, padx=(0, 12))

section_title(form_c, "👤  PATIENT DEMOGRAPHICS")
row1 = tk.Frame(form_c, bg=BG2); row1.pack(fill="x", pady=4)
tk.Label(row1, text="Age", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=0,padx=(0,8))
tk.Label(row1, text="City Tier", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=1,padx=8)
tk.Label(row1, text="Language", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=2,padx=8)

c_age      = styled_entry(row1, 14); c_age.insert(0,"40"); c_age.grid(row=1,column=0,padx=(0,8),pady=2)
c_tier     = styled_combo(row1,["Tier-1","Tier-2","Tier-3"],14); c_tier.grid(row=1,column=1,padx=8,pady=2)
c_lang     = styled_combo(row1,["Hindi","English","Tamil","Bengali","Telugu","Marathi"],14); c_lang.grid(row=1,column=2,padx=8,pady=2)

tk.Frame(form_c, bg=BORDER, height=1).pack(fill="x", pady=10)
section_title(form_c, "💻  PLATFORM & ACCESS")
row2 = tk.Frame(form_c, bg=BG2); row2.pack(fill="x", pady=4)
tk.Label(row2, text="Platform", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=0,padx=(0,8))
tk.Label(row2, text="Internet Quality", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=1,padx=8)
tk.Label(row2, text="Has Insurance", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=2,padx=8)

c_plat     = styled_combo(row2,["eSanjeevani","Practo","mFine","Other"],14); c_plat.grid(row=1,column=0,padx=(0,8),pady=2)
c_inet     = styled_combo(row2,["Good","Average","Poor"],14); c_inet.grid(row=1,column=1,padx=8,pady=2)
c_insur    = styled_combo(row2,["Yes","No"],14); c_insur.grid(row=1,column=2,padx=8,pady=2)

tk.Frame(form_c, bg=BORDER, height=1).pack(fill="x", pady=10)
section_title(form_c, "📋  CONSULTATION HISTORY")
row3 = tk.Frame(form_c, bg=BG2); row3.pack(fill="x", pady=4)
tk.Label(row3, text="# Consultations", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=0,padx=(0,8))
tk.Label(row3, text="Avg Wait (min)", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=1,padx=8)
tk.Label(row3, text="Satisfaction (1-5)", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=2,padx=8)

c_cons     = styled_entry(row3,14); c_cons.insert(0,"3"); c_cons.grid(row=1,column=0,padx=(0,8),pady=2)
c_wait     = styled_entry(row3,14); c_wait.insert(0,"15.0"); c_wait.grid(row=1,column=1,padx=8,pady=2)
c_sat      = styled_entry(row3,14); c_sat.insert(0,"3.5"); c_sat.grid(row=1,column=2,padx=8,pady=2)

row4 = tk.Frame(form_c, bg=BG2); row4.pack(fill="x", pady=4)
tk.Label(row4, text="Chronic Condition", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=0,padx=(0,8))
c_chronic  = styled_combo(row4,["No","Yes"],14); c_chronic.grid(row=1,column=0,padx=(0,8),pady=2)

# Right result panel
res_c = card(churn_main, padx=22, pady=18, width=300)
res_c.pack(side="right", fill="y", padx=(0,0))
res_c.pack_propagate(False)

tk.Label(res_c, text="🔮 Prediction Result", font=(FONT,11,"bold"), bg=BG2, fg=TEXT).pack(anchor="w", pady=(0,16))
tk.Frame(res_c, bg=BORDER, height=1).pack(fill="x", pady=(0,16))

churn_prob_var = tk.StringVar(value="—")
churn_label_var = tk.StringVar(value="Run prediction to see result")
churn_color_var = [MUTED]

prob_lbl  = tk.Label(res_c, textvariable=churn_prob_var, font=(FONT,36,"bold"), bg=BG2, fg=MUTED)
prob_lbl.pack(pady=(10,4))
tk.Label(res_c, text="Churn Probability", font=(FONT,9), bg=BG2, fg=MUTED).pack()

# Canvas gauge
gauge_canvas = tk.Canvas(res_c, width=240, height=24, bg=BG2, highlightthickness=0)
gauge_canvas.pack(pady=12)
gauge_canvas.create_rectangle(0,4,240,20, fill=BORDER, outline="")
gauge_fill_id = gauge_canvas.create_rectangle(0,4,0,20, fill=MUTED, outline="")

status_lbl = tk.Label(res_c, textvariable=churn_label_var, font=(FONT,9,"bold"),
                      bg=BG2, fg=MUTED, wraplength=240, justify="center")
status_lbl.pack(pady=8)

def validate_churn_inputs():
    try:
        age = int(c_age.get())
        if not (1 <= age <= 120):
            raise ValueError("Age must be between 1 and 120.")
        cons = int(c_cons.get())
        if cons < 0:
            raise ValueError("Consultations cannot be negative.")
        wait = float(c_wait.get())
        if wait < 0:
            raise ValueError("Wait time cannot be negative.")
        sat = float(c_sat.get())
        if not (1.0 <= sat <= 5.0):
            raise ValueError("Satisfaction must be between 1.0 and 5.0.")
        return True
    except ValueError as e:
        messagebox.showerror("Validation Error", str(e))
        return False

def run_churn():
    if not validate_churn_inputs():
        return
    try:
        patient = {
            "age": int(c_age.get()),
            "internet_quality": c_inet.get(),
            "city_tier": c_tier.get(),
            "language": c_lang.get(),
            "platform": c_plat.get(),
            "num_consultations": int(c_cons.get()),
            "avg_wait_time": float(c_wait.get()),
            "satisfaction_score": float(c_sat.get()),
            "has_insurance": 1 if c_insur.get() == "Yes" else 0,
            "chronic_condition": 1 if c_chronic.get() == "Yes" else 0,
        }
        result = predict_churn(patient)
        prob   = result["churn_probability"]
        pct    = prob * 100
        color  = DANGER if result["will_churn"] else SUCCESS
        churn_prob_var.set(f"{pct:.1f}%")
        prob_lbl.config(fg=color)
        gauge_canvas.itemconfig(gauge_fill_id, fill=color)
        gauge_canvas.coords(gauge_fill_id, 0, 4, int(240 * prob), 20)
        if result["will_churn"]:
            churn_label_var.set("⚠️ HIGH CHURN RISK\nConsider engagement interventions.")
            status_lbl.config(fg=DANGER)
        else:
            churn_label_var.set("✅ LOW CHURN RISK\nPatient likely to stay on platform.")
            status_lbl.config(fg=SUCCESS)
    except Exception as ex:
        messagebox.showerror("Prediction Error", str(ex))

def reset_churn():
    c_age.delete(0, tk.END); c_age.insert(0, "40")
    c_cons.delete(0, tk.END); c_cons.insert(0, "3")
    c_wait.delete(0, tk.END); c_wait.insert(0, "15.0")
    c_sat.delete(0, tk.END); c_sat.insert(0, "3.5")
    c_tier.current(0); c_lang.current(0); c_plat.current(0)
    c_inet.current(0); c_insur.current(0); c_chronic.current(0)
    churn_prob_var.set("—"); churn_label_var.set("Run prediction to see result")
    prob_lbl.config(fg=MUTED); status_lbl.config(fg=MUTED)
    gauge_canvas.coords(gauge_fill_id, 0, 4, 0, 20)
    gauge_canvas.itemconfig(gauge_fill_id, fill=MUTED)

btn_row_c = tk.Frame(form_c, bg=BG2); btn_row_c.pack(pady=(16,0), fill="x")
tk.Button(btn_row_c, text="🔮  Run Churn Prediction", font=(FONT,10,"bold"),
          bg=ACCENT, fg="#0a0f1e", relief="flat", bd=0,
          padx=20, pady=10, cursor="hand2", command=run_churn).pack(side="left", expand=True, fill="x", padx=(0,6))
tk.Button(btn_row_c, text="↺ Reset", font=(FONT,10,"bold"),
          bg=BG3, fg=MUTED, relief="flat", bd=0,
          padx=14, pady=10, cursor="hand2", command=reset_churn).pack(side="right")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
tab_risk = tk.Frame(nb, bg=BG)
nb.add(tab_risk, text="  ⚕️  Risk Classifier  ")

risk_main = tk.Frame(tab_risk, bg=BG)
risk_main.pack(fill="both", expand=True, padx=24, pady=20)

form_r = card(risk_main, padx=22, pady=18)
form_r.pack(side="left", fill="both", expand=True, padx=(0,12))

section_title(form_r, "🧑⚕️  PATIENT INFO", color=WARNING)
rr1 = tk.Frame(form_r, bg=BG2); rr1.pack(fill="x", pady=4)
tk.Label(rr1, text="Age", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=0,padx=(0,8))
tk.Label(rr1, text="Specialty", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=1,padx=8)
r_age      = styled_entry(rr1,14); r_age.insert(0,"50"); r_age.grid(row=1,column=0,padx=(0,8),pady=2)
r_spec     = styled_combo(rr1,["General","Cardiology","Dermatology","Pediatrics","Orthopedics"],14)
r_spec.grid(row=1,column=1,padx=8,pady=2)

tk.Frame(form_r, bg=BORDER, height=1).pack(fill="x", pady=10)
section_title(form_r, "💊  VITALS", color=WARNING)
rr2 = tk.Frame(form_r, bg=BG2); rr2.pack(fill="x", pady=4)
tk.Label(rr2, text="BP Systolic", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=0,padx=(0,8))
tk.Label(rr2, text="BP Diastolic", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=1,padx=8)
tk.Label(rr2, text="SpO2 (%)", font=(FONT,9), bg=BG2, fg=MUTED, width=16, anchor="w").grid(row=0,column=2,padx=8)
r_bps  = styled_entry(rr2,14); r_bps.insert(0,"120"); r_bps.grid(row=1,column=0,padx=(0,8),pady=2)
r_bpd  = styled_entry(rr2,14); r_bpd.insert(0,"80");  r_bpd.grid(row=1,column=1,padx=8,pady=2)
r_spo2 = styled_entry(rr2,14); r_spo2.insert(0,"97.0");r_spo2.grid(row=1,column=2,padx=8,pady=2)

tk.Frame(form_r, bg=BORDER, height=1).pack(fill="x", pady=10)
section_title(form_r, "🤒  SYMPTOMS", color=WARNING)
sym_frame = tk.Frame(form_r, bg=BG2); sym_frame.pack(fill="x", pady=4)
sym_vars = {}
symptoms = [("🌡️ Fever","fever"),("😷 Cough","cough"),("😴 Fatigue","fatigue"),
            ("🫁 Breathlessness","breath"),("💔 Chest Pain","chest")]
for i,(label,key) in enumerate(symptoms):
    v = tk.IntVar()
    sym_vars[key] = v
    cb = tk.Checkbutton(sym_frame, text=label, variable=v,
                        font=(FONT,10), bg=BG2, fg=TEXT,
                        selectcolor=BG3, activebackground=BG2,
                        activeforeground=TEXT)
    cb.grid(row=0, column=i, padx=10, pady=6, sticky="w")

# Right result
res_r = card(risk_main, padx=22, pady=18, width=300)
res_r.pack(side="right", fill="y")
res_r.pack_propagate(False)

tk.Label(res_r, text="⚕️ Risk Assessment", font=(FONT,11,"bold"), bg=BG2, fg=TEXT).pack(anchor="w", pady=(0,16))
tk.Frame(res_r, bg=BORDER, height=1).pack(fill="x", pady=(0,16))

risk_prob_var  = tk.StringVar(value="—")
risk_label_var = tk.StringVar(value="Enter vitals and click Assess Risk")
risk_prob_lbl  = tk.Label(res_r, textvariable=risk_prob_var, font=(FONT,36,"bold"), bg=BG2, fg=MUTED)
risk_prob_lbl.pack(pady=(10,4))
tk.Label(res_r, text="Risk Probability", font=(FONT,9), bg=BG2, fg=MUTED).pack()

risk_gauge = tk.Canvas(res_r, width=240, height=24, bg=BG2, highlightthickness=0)
risk_gauge.pack(pady=12)
risk_gauge.create_rectangle(0,4,240,20, fill=BORDER, outline="")
risk_fill_id = risk_gauge.create_rectangle(0,4,0,20, fill=MUTED, outline="")

risk_status_lbl = tk.Label(res_r, textvariable=risk_label_var, font=(FONT,9,"bold"),
                           bg=BG2, fg=MUTED, wraplength=240, justify="center")
risk_status_lbl.pack(pady=8)

# Vitals summary labels
tk.Frame(res_r, bg=BORDER, height=1).pack(fill="x", pady=8)
bp_status_var  = tk.StringVar(value="BP: —")
spo2_status_var= tk.StringVar(value="SpO2: —")
sym_status_var = tk.StringVar(value="Symptoms: —")
for var, icon in [(bp_status_var,"🩺"),(spo2_status_var,"💧"),(sym_status_var,"🤒")]:
    tk.Label(res_r, textvariable=var, font=(FONT,9), bg=BG2, fg=MUTED).pack(anchor="w", pady=2)

def validate_risk_inputs():
    try:
        age = int(r_age.get())
        if not (1 <= age <= 120):
            raise ValueError("Age must be between 1 and 120.")
        bps = int(r_bps.get())
        if not (60 <= bps <= 250):
            raise ValueError("BP Systolic must be between 60 and 250.")
        bpd = int(r_bpd.get())
        if not (40 <= bpd <= 150):
            raise ValueError("BP Diastolic must be between 40 and 150.")
        spo2 = float(r_spo2.get())
        if not (50.0 <= spo2 <= 100.0):
            raise ValueError("SpO2 must be between 50 and 100.")
        return True
    except ValueError as e:
        messagebox.showerror("Validation Error", str(e))
        return False

def run_risk():
    if not validate_risk_inputs():
        return
    try:
        bps  = int(r_bps.get())
        spo2_val = float(r_spo2.get())
        consultation = {
            "age": int(r_age.get()),
            "symptoms_fever": sym_vars["fever"].get(),
            "symptoms_cough": sym_vars["cough"].get(),
            "symptoms_fatigue": sym_vars["fatigue"].get(),
            "symptoms_breathlessness": sym_vars["breath"].get(),
            "symptoms_chest_pain": sym_vars["chest"].get(),
            "bp_systolic": bps,
            "bp_diastolic": int(r_bpd.get()),
            "spo2": spo2_val,
            "specialty": r_spec.get(),
        }
        result = predict_risk(consultation)
        prob   = result["risk_probability"]
        pct    = prob * 100
        color  = DANGER if result["high_risk"] else SUCCESS

        risk_prob_var.set(f"{pct:.1f}%")
        risk_prob_lbl.config(fg=color)
        risk_gauge.itemconfig(risk_fill_id, fill=color)
        risk_gauge.coords(risk_fill_id, 0, 4, int(240 * prob), 20)

        if result["high_risk"]:
            risk_label_var.set("🚨 HIGH RISK\nImmediate in-person consultation recommended.")
            risk_status_lbl.config(fg=DANGER)
        else:
            risk_label_var.set("✅ LOW RISK\nTelemedicine follow-up is appropriate.")
            risk_status_lbl.config(fg=SUCCESS)

        bp_s = "🔴 High" if bps > 140 else ("🟡 Borderline" if bps > 120 else "🟢 Normal")
        s2_s = "🔴 Low" if spo2_val < 95 else "🟢 Normal"
        sym_c = sum(v.get() for v in sym_vars.values())
        bp_status_var.set(f"🩺 Blood Pressure: {bps}/{r_bpd.get()} — {bp_s}")
        spo2_status_var.set(f"💧 SpO2: {spo2_val}% — {s2_s}")
        sym_status_var.set(f"🤒 Active Symptoms: {sym_c} / 5")

    except Exception as ex:
        messagebox.showerror("Prediction Error", str(ex))

def reset_risk():
    r_age.delete(0, tk.END); r_age.insert(0, "50")
    r_bps.delete(0, tk.END); r_bps.insert(0, "120")
    r_bpd.delete(0, tk.END); r_bpd.insert(0, "80")
    r_spo2.delete(0, tk.END); r_spo2.insert(0, "97.0")
    r_spec.current(0)
    for v in sym_vars.values(): v.set(0)
    risk_prob_var.set("—"); risk_label_var.set("Enter vitals and click Assess Risk")
    risk_prob_lbl.config(fg=MUTED); risk_status_lbl.config(fg=MUTED)
    risk_gauge.coords(risk_fill_id, 0, 4, 0, 20)
    risk_gauge.itemconfig(risk_fill_id, fill=MUTED)
    bp_status_var.set("BP: —"); spo2_status_var.set("SpO2: —"); sym_status_var.set("Symptoms: —")

btn_row_r = tk.Frame(form_r, bg=BG2); btn_row_r.pack(pady=(16,0), fill="x")
tk.Button(btn_row_r, text="⚕️  Assess Clinical Risk", font=(FONT,10,"bold"),
          bg=WARNING, fg="#0a0f1e", relief="flat", bd=0,
          padx=20, pady=10, cursor="hand2", command=run_risk).pack(side="left", expand=True, fill="x", padx=(0,6))
tk.Button(btn_row_r, text="↺ Reset", font=(FONT,10,"bold"),
          bg=BG3, fg=MUTED, relief="flat", bd=0,
          padx=14, pady=10, cursor="hand2", command=reset_risk).pack(side="right")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PATIENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
tab_data = tk.Frame(nb, bg=BG)
nb.add(tab_data, text="  🗃️  Patient Data  ")

data_top = tk.Frame(tab_data, bg=BG)
data_top.pack(fill="x", padx=20, pady=(16,8))
tk.Label(data_top, text="🗃️  Patient Explorer", font=(FONT,13,"bold"), bg=BG, fg=TEXT).pack(side="left")

# Treeview
tree_frame = tk.Frame(tab_data, bg=BG)
tree_frame.pack(fill="both", expand=True, padx=20, pady=(0,16))

cols = ("Age","City Tier","Platform","Internet","Satisfaction","Wait(min)","Insurance","Churned")
tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=22)

style.configure("Treeview", background=BG2, foreground=TEXT,
                fieldbackground=BG2, rowheight=28, font=(FONT,9))
style.configure("Treeview.Heading", background=BG3, foreground=ACCENT,
                font=(FONT,9,"bold"), relief="flat")
style.map("Treeview", background=[("selected","#1e3a5f")], foreground=[("selected",TEXT)])

for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=110, anchor="center")

vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
vsb.pack(side="right", fill="y")
hsb.pack(side="bottom", fill="x")
tree.pack(fill="both", expand=True)

try:
    import pandas as pd
    df = pd.read_csv("data/patients.csv")
    for _, row in df.head(150).iterrows():
        churned = "⚠️ Yes" if row["churned"] == 1 else "✅ No"
        insured = "✅ Yes" if row["has_insurance"] == 1 else "❌ No"
        tree.insert("", "end", values=(
            row["age"], row["city_tier"], row["platform"],
            row["internet_quality"], row["satisfaction_score"],
            row["avg_wait_time"], insured, churned
        ))
    tk.Label(data_top, text=f"Showing 150 of {len(df):,} records",
             font=(FONT,9), bg=BG, fg=MUTED).pack(side="right")
except Exception:
    tk.Label(data_top, text="No data — run model.py first",
             font=(FONT,9), bg=BG, fg=DANGER).pack(side="right")

# ── Status bar ────────────────────────────────────────────────────────────────
status_bar = tk.Frame(root, bg="#0d1f3c", height=28)
status_bar.pack(fill="x", side="bottom")
tk.Label(status_bar, text="🟢  Models loaded  |  MediAI v1.0  |  India Telemedicine Platform",
         font=(FONT,8), bg="#0d1f3c", fg=MUTED).pack(side="left", padx=16)

root.mainloop()
