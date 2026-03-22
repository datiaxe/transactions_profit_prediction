import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os

# ── path relativ la radacina proiectului ──────────────────────────────────────
MODEL_PATH ='../models/profit_prediction_engine_HGBRegressor_final.pkl'

# ── mapare sub-categorie → categorie ─────────────────────────────────────────
CAT_MAP = {
    "Chairs": "Furniture", "Tables": "Furniture", "Bookcases": "Furniture",
    "Furnishings": "Furniture", "Supplies": "Office Supplies",
    "Storage": "Office Supplies", "Binders": "Office Supplies",
    "Paper": "Office Supplies", "Labels": "Office Supplies",
    "Envelopes": "Office Supplies", "Fasteners": "Office Supplies",
    "Art": "Office Supplies", "Phones": "Technology", "Copiers": "Technology",
    "Machines": "Technology", "Accessories": "Technology", "Appliances": "Technology",
}

SUBCATS  = sorted(CAT_MAP.keys())
REGIONS  = ["Central", "East", "South", "West"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SHIPS    = ["Standard Class", "Second Class", "First Class", "Same Day"]
SEG_ENC  = {s: i for i, s in enumerate(SEGMENTS)}

model_pack = None


def load_model():
    global model_pack
    try:
        model_pack = joblib.load(MODEL_PATH)
        return "✅ Model incarcat cu succes."
    except FileNotFoundError:
        return f"❌ Fisierul nu a fost gasit:\n{MODEL_PATH}"
    except Exception as e:
        return f"❌ Eroare: {e}"


def predict(sales, discount, quantity,
            sub_category, region,
            cust_avg_margin, cust_loss_rate, cust_avg_profit,
            p_last, customer_segment):

    if model_pack is None:
        return "❌ Apasa mai intai 'Incarca Model'.", "", "", ""

    # ── lookups din model ─────────────────────────────────────────────────────
    rsr = model_pack["risk_subcat_region"]
    rc  = model_pack["risk_category"]
    sp  = model_pack["ship_profit"]
    sr  = model_pack["seasonal_risk"]

    category = CAT_MAP.get(sub_category, "Office Supplies")

    row_rsr  = rsr[(rsr["sub-category"] == sub_category) & (rsr["region"] == region)]
    risk_sr  = float(row_rsr["risk_subcat_region"].values[0]) if len(row_rsr) else 0.0

    row_rc   = rc[rc["category"] == category]
    risk_cat = float(row_rc["risk_category"].values[0]) if len(row_rc) else 0.0

    row_sp   = sp[sp["ship_mode"] == "Standard Class"]
    ship_avg = float(row_sp["ship_profit_avg"].values[0]) if len(row_sp) else 55.0

    row_seas = sr[(sr["order_month"] == 6) & (sr["sub-category"] == sub_category)]
    seas     = float(row_seas["seasonal_risk"].values[0]) if len(row_seas) else 0.0

    p_margin_est = float(cust_avg_margin)

    # ── construire rand ───────────────────────────────────────────────────────
    row = {
        "sales":              float(sales),
        "discount":           float(discount),
        "quantity":           int(quantity),
        "log_sales":          np.log1p(float(sales)),
        "unit_sales":         float(sales) / (int(quantity) + 1e-5),
        "discount_sq":        float(discount) ** 2,
        "high_disc_pct":      1.0 if float(discount) >= 0.3 else 0.0,
        "zero_disc_pct":      1.0 if float(discount) == 0.0 else 0.0,
        "high_discount":      1 if float(discount) >= 0.3 else 0,
        "zero_discount":      1 if float(discount) == 0.0 else 0,
        "risk_subcat_region": risk_sr,
        "risk_category":      risk_cat,
        "risk_product":       0.0,
        "seasonal_risk":      seas,
        "cust_total_profit":  float(cust_avg_profit) * 20,
        "cust_avg_profit":    float(cust_avg_profit),
        "cust_avg_margin":    float(cust_avg_margin),
        "cust_n_orders":      20,
        "cust_loss_rate":     float(cust_loss_rate),
        "cust_recency":       1,
        "cust_tenure":        3,
        "cust_has_history":   1,
        "cust_high_disc_rate": 0.05,
        "is_high_season":     0,
        "is_q1":              0,
        "shipping_delay":     2.5,
        "ship_profit_avg":    ship_avg,
        "ship_mode_enc":      0,
        "segment_enc":        SEG_ENC.get(customer_segment, 0),
        "p_last":             float(p_last),
        "profit_last":        float(p_last) * float(sales) * 0.8,
        "margin_trend":       p_margin_est - float(p_last),
        "discount_efficiency": float(sales) * p_margin_est / (float(discount) + 0.01),
        "high_loss_risk":     1 if p_margin_est < -0.2 else 0,
        "n_orders":           5,
        "n_products":         3,
        "product_diversity":  3 / (5 + 1e-5),
        "Art_West":           1 if (category == "Furniture" and region == "West") else 0,
        "Acc_East":           1 if (sub_category == "Accessories" and region == "East") else 0,
    }

    df = pd.DataFrame([row])

    # ── clustering ────────────────────────────────────────────────────────────
    clust_input = pd.DataFrame(
        [[float(sales), p_margin_est, float(discount),
          float(cust_avg_margin), float(cust_loss_rate)]],
        columns=["sales", "p_margin", "discount", "cust_avg_margin", "cust_loss_rate"]
    )
    X_cl    = model_pack["robust_scaler"].transform(clust_input)
    segment = model_pack["kmeans_model"].predict(X_cl)[0]
    df["segment_client"] = segment

    pt     = model_pack["power_transformer"]
    alpha  = model_pack["blend_alpha"]
    feats  = model_pack["features_list"]
    mfeats = model_pack["meta_features"]

    # ── expert ────────────────────────────────────────────────────────────────
    if segment in model_pack["experts"]:
        pred_ex = model_pack["experts"][segment].predict(df[feats])
        marg_ex = pt.inverse_transform(
            pd.DataFrame(pred_ex, columns=["margin_w"])
        ).ravel()
        ep = marg_ex[0] * float(sales)
    else:
        marg_ex = np.array([0.0])
        ep      = 0.0
        alpha   = 0.0

    # ── meta-model ────────────────────────────────────────────────────────────
    pred_mt = model_pack["meta_model"].predict(df[mfeats])
    marg_mt = pt.inverse_transform(
        pd.DataFrame(pred_mt, columns=["margin_w"])
    ).ravel()
    mp = marg_mt[0] * float(sales)

    # ── blend final ───────────────────────────────────────────────────────────
    fp = alpha * ep + (1 - alpha) * mp
    fm = alpha * marg_ex[0] + (1 - alpha) * marg_mt[0]

    icon        = "🟢" if fp >= 0 else "🔴"
    out_final   = f"{icon}  Profit: ${fp:,.2f}   |   Marja: {fm*100:.2f}%"
    out_segment = f"Segment KMeans: {segment}   |   Categorie: {category}   |   Risk subcat×reg: ${risk_sr:.1f}"
    out_expert  = f"Expert  → ${ep:,.2f}   (marja {marg_ex[0]*100:.2f}%)"
    out_meta    = f"Meta    → ${mp:,.2f}   (marja {marg_mt[0]*100:.2f}%)   |   α={alpha:.2f}"

    return out_final, out_segment, out_expert, out_meta


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="Profit Prediction Engine", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 📦 Profit Prediction Engine")
    gr.Markdown("Completeaza campurile si apasa **Predict**. Restul valorilor sunt implicite.")

    with gr.Row():
        btn_load    = gr.Button("📂 Incarca Model", variant="primary", scale=1)
        load_status = gr.Textbox(label="Status", interactive=False, scale=4)
    btn_load.click(fn=load_model, outputs=load_status)

    gr.Markdown("---")
    gr.Markdown("### 🛒 Tranzactie")
    sales    = gr.Number(label="Sales ($)",  value=1500.0, precision=2)
    discount = gr.Slider(label="Discount",   value=0.15, minimum=0.0, maximum=0.8, step=0.05)
    quantity = gr.Number(label="Quantity",   value=50, precision=0)

    gr.Markdown("### 📍 Produs & Locatie")
    sub_category = gr.Dropdown(label="Sub-categorie", choices=SUBCATS,  value="Phones")
    region       = gr.Dropdown(label="Regiune",       choices=REGIONS,  value="East")

    gr.Markdown("### 👤 Profil Client (RFM)")
    cust_avg_margin = gr.Slider(label="Marja medie istorica",   value=0.18, minimum=-0.5, maximum=0.8, step=0.01)
    cust_loss_rate  = gr.Slider(label="Rata pierdere istorica", value=0.10, minimum=0.0,  maximum=1.0, step=0.01)
    cust_avg_profit = gr.Number(label="Profit mediu / rand ($)", value=120.0, precision=2)

    gr.Markdown("### 🕐 Temporal & Segment")
    p_last           = gr.Slider(label="Marja perioadei anterioare (p_last)", value=0.15, minimum=-1.0, maximum=1.0, step=0.01)
    customer_segment = gr.Dropdown(label="Segment client", choices=SEGMENTS, value="Consumer")

    gr.Markdown("---")
    btn_predict = gr.Button("🔮 Predict", variant="primary", size="lg")

    gr.Markdown("### 📈 Rezultat")
    out_final   = gr.Textbox(label="PREDICTIE FINALA",   interactive=False)
    out_segment = gr.Textbox(label="Detalii segment",    interactive=False)
    out_expert  = gr.Textbox(label="Expert per-segment", interactive=False)
    out_meta    = gr.Textbox(label="Meta-model + blend", interactive=False)

    btn_predict.click(
        fn=predict,
        inputs=[sales, discount, quantity,
                sub_category, region,
                cust_avg_margin, cust_loss_rate, cust_avg_profit,
                p_last, customer_segment],
        outputs=[out_final, out_segment, out_expert, out_meta]
    )

if __name__ == "__main__":
    demo.launch()
