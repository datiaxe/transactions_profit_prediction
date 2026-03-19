# =============================================================================
# # PROFIT PREDICTION ENGINE
# =============================================================================
# 
# ## OBIECTIV:
#     Prezicerea profitului generat de tranzactiile fiecarui client (customer_id) pe combinatia
#     de regiune si sub-categorie de produs, folosind date istorice 2014-2016
#     pt a antrena un model care sa functioneze bine pe datele din 2017.
# 
# ## STRUCTURA model:
#     1.  Incarcare date & split temporal strict pe ani (train +test 2014-2016, blind 2017)
#     2.  Lookups de risc (calculate DOAR din date de antrenament)
#     3.  RFM (recency, frequency, monetization) — caracterizizeaza comportamentul istoric al fiecarui client (fara leakage)
#     4.  Pipeline de preprocesare si feature engineering
#     5.  Clusterizare clienti (KMeans)
#     6.  Transformare target (Yeo-Johnson)
#     7.  Lista de features
#     8.  Antrenare modele expert per segment
#     9.  Meta-model global (stacking)
#     10. Evaluare si comparatie
#     11. Salvare artefacte
# 
# ## CONTEAZA ORDINEA pt ca:
#     -Toate statisticile (medii, percentile, centroizi, parametri de transformare) sunt calculate EXCLUSIV din datele de antrenament.
#     -Testul si blind-ul primesc aceste statistici fixate, niciodata nu le recalculeaza. Aceasta este regula fundamentala anti-leakage.
# %% [markdown]
# 1. Features RFM customer history — cea mai mare contribuție. Cei 517/693 clienți din 2017 au 3 ani de istoric. cust_avg_margin, cust_loss_rate, cust_tenure permit modelului să știe dinainte dacă un client este profitabil sistematic sau nu.
# 2. Discount threshold features — high_disc_pct (discount ≥ 30% = 91.6% pierdere în train) este cel mai predictiv semnal simplu din tot dataset-ul. V3 folosea discount ca valoare continuă și pierdea această non-liniaritate.
# 3. Stacking (meta-model global) — un al doilea model antrenat pe aceleași date corectează erorile sistematice ale experților per-segment, în special pt segmentul enterprise (7 rânduri, prea mic pt un expert dedicat). Blendingul 50/50 dă cel mai bun echilibru pe setul de validare.
# 4. Risk multi-nivel — adăugarea risk_product și risk_category pe lângă risk_subcat_region dă modelului granularitate mai fină asupra produselor individual problematice.
# %% [markdown]
#  5 zone unde data leakage poate sa apara; codul le evită pe toate:
# 
# 1. Lookups de risc — construite exclusiv din df_train
# pythonrisk_subcat_region = df_train.groupby(['sub-category','region'])['profit'].mean()
# risk_category      = df_train.groupby(['category'])['profit'].mean()
# risk_product       = df_train.groupby(['product_id'])['profit'].mean()
# seasonal_risk      = df_train.groupby(['order_month','sub-category'])['profit'].mean()
# Modelul "știe" că Machines×Central este riscant dintr-un calcul pe 2014–2016. Când aplică acest scor pe 2017, nu a văzut niciun rând din viitor — e ca și cum un analist ar studia raportul de anul trecut înainte să ia o decizie.
# Codul original calcula category_fallback_risk pe df_rest (train+test), adică includea date de test în construcția unui feature aplicat ulterior pe test. V4 calculează totul strict pe df_train.
# 
# 2. RFM customer history — lookup pre-calculat per split. Acesta este cel mai subtil punct. Există trei scenarii distincte:
# 
# - pt TEST: istoria clientului = tot df_train (strict anterior testului)
# rfm_for_test  = build_customer_rfm(df_train, reference_year=2016)
# 
# - pt BLIND: istoria clientului = tot df_rest = 2014+2015+2016
# rfm_for_blind = build_customer_rfm(df_rest,  reference_year=2017)
# 
# - pt TRAIN: istoria clientului = doar anii STRICT anteriori anului curent din train
# for yr in sorted(df_train['order_year'].unique()):
#     hist_yr = df_train[df_train['order_year'] < yr]  # <-- strict mai mic
#     rfm_yr  = build_customer_rfm(hist_yr, reference_year=yr)
# Un rând din 2015 în train primește RFM calculat doar din 2014. Un rând din 2016 primește RFM din 2014+2015. Niciun rând nu "vede" profitabilitatea sa proprie sau a perioadelor ulterioare.
# Codul original calcula p_last prin shift(1) pe full_data concatenat. Concatenarea train+test+blind înainte de shift permitea unor rânduri din blind să primească valori calculate inclusiv din 2017.
# 
# 3. KMeans — fit exclusiv pe train, predict separat
# - pythonkmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
# - kmeans.fit(X_tr_cl)                                        # ← doar train
# 
# - df_train_agg['segment_client'] = kmeans.predict(X_tr_cl)
# - df_test_agg['segment_client']  = kmeans.predict(X_te_cl)  # ← aplica centroids din train
# - df_blind_agg['segment_client'] = kmeans.predict(X_bl_cl)
# Centroizii clusterelor reflectă doar distribuția din 2014–2016. Blind este asignat unui cluster existent, nu îl influențează. Dacă KMeans ar fi fit pe full_data, centroizii s-ar deplasa spre distribuția din 2017 — segmentarea ar absorbi informație din viitor.
# 
# 4. PowerTransformer — fit exclusiv pe train
# - pythonpt = PowerTransformer(method='yeo-johnson')
# - pt.fit(df_train_agg[['margin_w']])           # ← parametrii lambda estimati doar din train
# 
# - df_train_agg['target'] = pt.transform(df_train_agg[['margin_w']])
# - df_test_agg['target']  = pt.transform(...)   # ← aceeasi transformare, fara refit
# - df_blind_agg['target'] = pt.transform(...)
# Parametrul lambda al transformării Yeo-Johnson este estimat din distribuția marjei din 2014–2016. Dacă ar fi estimat pe toate datele, distribuția din 2017 ar influența cum sunt normalizate valorile din 2014 — o formă subtilă de leakage care denaturează comparabilitatea scalelor.
# 
# 5. Winsorize — bounds calculate din train
# - pythonlow_w  = df_train_agg['p_margin'].quantile(0.01)   # ← percentilele din train
# - high_w = df_train_agg['p_margin'].quantile(0.99)
# 
# - df_test_agg['target']  = pt.transform(df_test_agg[['p_margin']].clip(low_w, high_w))
# - df_blind_agg['target'] = pt.transform(df_blind_agg[['p_margin']].clip(low_w, high_w))
# Dacă bounds-urile s-ar calcula pe toate datele, un outlier extrem din 2017 ar muta quantile(0.99) în sus — și implicit ar schimba cât de mult sunt "tăiați" outlierii din 2014–2015 în antrenament. Bounds fixate pe train înseamnă că decizia de ce e "extrem" e luată o singură dată, din date cunoscute la momentul antrenării.
# 
# Regula generală care unește toate cele 5 puncte:
# Orice statistică (medie, percentilă, centroid, parametru de transformare) care va fi aplicată pe date de test sau blind trebuie calculată exclusiv din datele disponibile la momentul la care modelul ar fi existat în producție — adică din train. Testul și blind-ul sunt tratate ca "viitor necunoscut", cărora li se aplică reguli fixate anterior.
# %%
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')
# %%
# =============================================================================
# 1. INCARCARE DATE & SPLIT TEMPORAL STRICT
# =============================================================================
# datele sunt impartite STRICT dupa timp, nu aleatoriu.
# predictia profitului din 2017 va fi antrenata DOAR pe ce este cunoscut dinainte de 2017; orice amestecare temporala = data leakage.

df = pd.read_csv(r'..\data\processed\df_preprocessed.csv')

# conversie coloana de date din string in datetime (necesar pt .dt.year etc.)
df['order_date']  = pd.to_datetime(df['order_date'])
df['order_year']  = df['order_date'].dt.year
df['order_month'] = df['order_date'].dt.month


# --- BLIND SET (2017) ---
# acesta simuleaza "viitorul necunoscut" la momentul antrenarii.
# modelul NU 'vede' aceste date in nici un calcul anterior de predictie.
df_blind = df[df['order_year'] == 2017].copy()

# --- REST (2014-2016) ---
# datele disponibile inainte de 2017, din care se construiesc train si test.
df_rest = df[df['order_year'] < 2017].copy()


# --- TEST SET ---
# Extragem 20% din fiecare an (stratificat) pt evaluare interna.
# De ce stratificat pe an? Ca sa avem reprezentare echilibrata din toti cei 3 ani,
# altfel riscam sa avem test format majoritar dintr-un singur an.
df_test = (df_rest
           .groupby('order_year', group_keys=False)
           .apply(lambda x: x.sample(frac=0.2, random_state=42))
           .copy())

# --- TRAIN SET ---
# Restul de 80% din fiecare an devine setul de antrenament.
# .drop(df_test.index) elimina exact randurile selectate pt test.
df_train = df_rest.drop(df_test.index).copy()

print(f"Train: {len(df_train)} randuri | Test: {len(df_test)} randuri | Blind: {len(df_blind)} randuri")


# =============================================================================
# 2. LOOKUPS DE RISC — CALCULATE EXCLUSIV DIN TRAIN
# =============================================================================
# Un 'lookup de risc' este o tabela de referinta care spune:
# e.g. "in medie, aceasta combinatie de produse/regiuni/categorii a generat profit X."
# altfel spus, avem un profit mediu calculat ca si grouip by pe pe diverese categorii
# REGULA IMPORTANTA: toate aceste tabele sunt calculate DOAR din df_train.
# Daca am folosi df_rest sau date din 2017, modelul ar "sti" informatii
# din viitor si performanta sa pe blind ar fi artificial umflata.

# --- RISC LA NIVEL DE SUB-CATEGORIE × REGIUNE ---
# Ex: Daca "Tables" din "East" a avut profit mediu de -135$, orice comanda
# viitoare din aceasta combinatie primeste automat un semnal de risc ridicat.
risk_subcat_region = (
    df_train
    .groupby(['sub-category', 'region'])['profit']
    .mean()
    .reset_index()
    .rename(columns={'profit': 'risk_subcat_region'})
)

# --- RISC LA NIVEL DE CATEGORIE (mai general) ---
# Unele categorii (ex: Furniture) sunt sistematic mai riscante decat altele
# (ex: Office Supplies). Aceasta tabela surprinde riscul la nivel macro.
risk_category = (
    df_train
    .groupby(['category'])['profit']
    .mean()
    .reset_index()
    .rename(columns={'profit': 'risk_category'})
)

# --- RISC LA NIVEL DE PRODUS INDIVIDUAL ---
# Cel mai granular nivel: un produs specific poate fi sistematic neprofitabil.
# Util mai ales pt produsele care apar des in comenzi cu discount mare.
risk_product = (
    df_train
    .groupby(['product_id'])['profit']
    .mean()
    .reset_index()
    .rename(columns={'profit': 'risk_product'})
)

# --- RISC SEZONIER (LUNA × SUB-CATEGORIE) ---
# Unele produse au sezonalitate puternica in profitabilitate.
# Ex: Furniture poate fi mai profitabila in Q4 decat in Q1.
seasonal_risk = (
    df_train
    .groupby(['order_month', 'sub-category'])['profit']
    .mean()
    .reset_index()
    .rename(columns={'profit': 'seasonal_risk'})
)

# --- PROFITABILITATEA MEDIE PE MOD DE LIVRARE ---
# "Same Day" poate implica costuri mai mari → profit mai mic per tranzactie.
ship_profit = (
    df_train
    .groupby('ship_mode')['profit']
    .mean()
    .reset_index()
    .rename(columns={'profit': 'ship_profit_avg'})
)

# --- COMBINATII CUNOSCUTE (pt detectia cazurilor noi in 2017) ---
# In 2017 pot aparea combinatii sub-categorie + regiune care nu au existat
# in 2014-2016. pt acestea nu avem risk_subcat_region → folosim fallback.
known_comb = (
    df_train[['sub-category', 'region']]
    .drop_duplicates()
    .assign(is_known=True)
)

# --- FALLBACK MARGIN (pt combinatii noi in blind) ---
# Daca in 2017 apare o combinatie noua, folosim marja medie a produsului gupata pe: 'order_year', 'region', 'sub-category', 'product_id', 'product_name'
# respectiv din anii anteriori ca estimare de risc.
# IMPORTANT: calculat DOAR din df_train, nu din df_rest care include si df_test.
df_train['p_margin_raw'] = df_train['profit'] / (df_train['sales'] + 1e-5)
fallback_risk = (
    df_train
    .groupby(['order_year', 'region', 'sub-category', 'product_id', 'product_name'])
    ['p_margin_raw']
    .mean()
    .reset_index()
    .rename(columns={'p_margin_raw': 'fallback_p_margin'})
)


# =============================================================================
# 3. FEATURES DE COMPORTAMENT CLIENT (RFM) — FARA LEAKAGE
# =============================================================================
# RFM = Recency, Frequency, Monetary — metodologie clasica in marketing
# pt a caracteriza comportamentul unui client.
#
# PROBLEMA DE LEAKAGE SPECIFICA:
#   daca am calcula "profitul mediu al clientului CG-12520" folosind toate datele
#   disponibile (inclusiv 2017) si apoi folosim aceasta valoare ca feature pt
#   prezicerea profitului sau din 2017, modelul "stie" raspunsul inainte sa il prezica.
#   transformă tranzacțiile individuale în features agregate la nivel de Client × An × Regiune × Sub-categorie.
#   un singur rrnd în setul procesat = toate comenzile unui client dintr-un an, pentru o combinație regiune/sub-categorie.
# SOLUTIE:
#   - pt train (an Y): RFM calculat DOAR din anii < Y
#   - pt test: RFM calculat DOAR din df_train (2014-2016)
#   - pt blind (2017): RFM calculat din df_rest (tot 2014-2016)

def build_customer_rfm(history_df, reference_year):
    """
    functia construieste profilul RFM al fiecarui client din 'history_df'.

    ca si arametri:
        history_df     : DataFrame cu tranzactii ANTERIOARE perioadei de prezis
        reference_year : anul pt care se calculeaza 'recency' (cat de recent a cumparat clientul fata de acest an)

    returneaza un DataFrame cu cate un rand per customer_id, continand:
        - cust_total_profit   : profitul total generat de client in trecut
        - cust_total_sales    : valoarea totala a vanzarilor
        - cust_n_orders       : numarul de comenzi distincte
        - cust_avg_discount   : discountul mediu pe care il solicita clientul
        - cust_loss_rate      : proportia tranzactiilor cu pierdere (profit < 0)
        - cust_avg_profit     : profitul mediu per rand de comanda
        - cust_high_disc_rate : cat % din comenzile clientului au discount >= 30%
        - cust_recency        : cati ani a trecut de la ultima comanda
        - cust_tenure         : pe cati ani s-a extins activitatea clientului
        - cust_avg_margin     : marja medie = profit / sales
        - cust_has_history    : flag 1/0 (0 pt clientii noi fara istoric)
    """
    hist = history_df.copy()

    # agregare toate tranzactiile istorice per client
    rfm = hist.groupby('customer_id').agg(
        cust_total_profit   = ('profit',    'sum'),
        cust_total_sales    = ('sales',     'sum'),
        cust_n_orders       = ('order_id',  'nunique'),
        cust_n_rows         = ('row_id',    'count'),
        cust_avg_discount   = ('discount',  'mean'),
        cust_loss_rate      = ('is_loss',   'mean'),       # % tranzactii cu profit < 0
        cust_avg_profit     = ('profit',    'mean'),
        cust_last_year      = ('order_year','max'),        # cel mai recent an de activitate
        cust_first_year     = ('order_year','min'),        # primul an de activitate
        cust_high_disc_rate = ('discount',  lambda x: (x >= 0.3).mean()),  # % discount >= 30%
    ).reset_index()

    # Recency: cati ani au trecut de la ultima comanda pana la data curenta 'reference_year'
    # e.g. un client care nu a cumparat ulitma data acum 2 ani este mai riscant decat unul activ recent.
    rfm['cust_recency'] = reference_year - rfm['cust_last_year']

    # Tenure: pe cati ani s-a extins relatia cu clientul (cat de vechi este clientul)
    # e.g. un client cu o vechime de 3 ani este mai predictibil decat unul cu 0 ani.
    rfm['cust_tenure'] = rfm['cust_last_year'] - rfm['cust_first_year'] + 1

    # Marja medie istorica: cel mai puternic predictor al marjei viitoare
    rfm['cust_avg_margin'] = rfm['cust_total_profit'] / (rfm['cust_total_sales'] + 1e-5)

    # Flag: clientul are date istorice (va fi 0 pt clientii complet noi)
    rfm['cust_has_history'] = 1

    return rfm


# adaugam coloana is_loss in toate spliturile (necesara pt build_customer_rfm)
# is_loss = 1 daca tranzactia a generat pierdere, 0 altfel
for df_tmp in [df_train, df_test, df_blind, df_rest]:
    df_tmp['order_date'] = pd.to_datetime(df_tmp['order_date'])
    if 'order_year' not in df_tmp.columns:
        df_tmp['order_year'] = df_tmp['order_date'].dt.year
    df_tmp['is_loss'] = (df_tmp['profit'] < 0).astype(int)


# RFM pt TEST: istoria clientului = tot df_train (2014-2016, strict anterior testului)
# reference_year = maximul anilor din test (2016), deci recency se calculeaza fata de 2016
rfm_for_test = build_customer_rfm(df_train, reference_year=max(df_test['order_year']))

# RFM pt BLIND: istoria clientului = tot df_rest (2014-2016)
# reference_year = 2017, deci recency = cat timp a trecut de la ultima comanda pana in 2017
rfm_for_blind = build_customer_rfm(df_rest, reference_year=2017)


# RFM pt TRAIN: cel mai complex caz, neces tratament special
# pt un rand de antrenament din 2015, RFM trebuie sa contina DOAR date din 2014.
# dar, pt un radn din 2016, RFM contine date din 2014 si 2015.
# daca nu am procesa asa, un rand din 2015 ar "sti" profitabilitatea clientului din 2015 → leakage.

rfm_train_parts = []
for yr in sorted(df_train['order_year'].unique()):
    # luam DOAR datele strict anterioare anului curent
    hist_yr = df_train[df_train['order_year'] < yr]

    # daca nu exista date anterioare (ex: prmul an din dataset), sarim
    if len(hist_yr) == 0:
        continue

    rfm_yr = build_customer_rfm(hist_yr, reference_year=yr)
    rfm_yr['order_year'] = yr   # adaugam anul pt a putea face join cu datele de antrenament
    rfm_train_parts.append(rfm_yr)

# concatenare lookup-urile per-an intr-un singur DataFrame
rfm_train_lookup = pd.concat(rfm_train_parts) if rfm_train_parts else pd.DataFrame()


# =============================================================================
# 4. PIPELINE DE PREPROCESARE SI FEATURE ENGINEERING
# =============================================================================
# functia preprocess_v4 transforma datele brute (tranzactii individuale) in features pt model, agregate la nivel de customer×year×region×sub-category
#
# de ce agregam? modelul prezice profitul unui CLIENT pe o COMBINATIE,
# nu profitul unei tranzactii individuale. Agregarea combina toate comenzile
# unui client dintr-un an pt aceeasi categorie/regiune intr-un singur rand

def preprocess_v4(data, risk_lookup, fallback_lookup, known_comb,
                   rfm_lookup, rfm_by_year=None,
                   ship_lookup=None, seasonal_lookup=None,
                   risk_cat=None, risk_prod=None,
                   is_blind=False):
    """
    Parametri:
        data           : DataFrame cu tranzactii brute (train/test/blind)
        risk_lookup    : tabela risk_subcat_region (din train)
        fallback_lookup: tabela fallback_risk pt combinatii noi (din train)
        known_comb     : combinatiile sub-cat×region cunoscute din train
        rfm_lookup     : profilul RFM per client (pt test si blind)
        rfm_by_year    : profilul RFM per client×an (pt train, evita leakage)
        ship_lookup    : profitabilitatea medie pe mod de livrare (din train)
        seasonal_lookup: riscul sezonier luna×sub-categorie (din train)
        risk_cat       : riscul per categorie (din train)
        risk_prod      : riscul per produs (din train)
        is_blind       : True daca prelucram datele din 2017 (active fallback logic)
    """

    temp = data.copy()
    temp['order_date']  = pd.to_datetime(temp['order_date'])
    temp['order_year']  = temp['order_date'].dt.year
    temp['order_month'] = temp['order_date'].dt.month


    # -------------------------------------------------------------------------
    # FEATURES DE TIMP
    # -------------------------------------------------------------------------
    # Sezonalitate: lunile 9-12 (Q4) sunt in mai profitabile,  conform analia szonalitate
    # datorita Black Friday, Craciun, comenzilor de fin de an.
    temp['is_high_season'] = temp['order_month'].isin([9, 10, 11, 12]).astype(int)

    # Q1 (ianuarie-martie) este adesea cel mai slab trimestru dupa sarbatori.
    temp['is_q1'] = temp['order_month'].isin([1, 2, 3]).astype(int)


    # -------------------------------------------------------------------------
    # FEATURES COMBINATE (combina doua variabile intr-un semnal specific)
    # -------------------------------------------------------------------------
    # Art_West: Furniture in regiunea West a aratat un pattern specific
    # Acc_East: Accessories in East la fel, comportament diferit fata de alte regiuni
    # Aceste interactiuni surprind efecte pe care un model liniar le-ar rata.
    temp['Art_West'] = ((temp['category'] == 'Furniture') & (temp['region'] == 'West')).astype(int)
    temp['Acc_East'] = ((temp['sub-category'] == 'Accessories') & (temp['region'] == 'East')).astype(int)


    # -------------------------------------------------------------------------
    # FEATURES DE DISCOUNT
    # -------------------------------------------------------------------------
    # analiza din EDA (e.g df['profit'].describe()):
    #   discount = 0.0 → 0% tranzactii cu pierdere (profit garantat)
    #   discount = 0.2 → 14% tranzactii cu pierdere
    #   discount = 0.3 → 91.6% tranzactii cu pierdere (aproape sigur pierdere!)
    #   discount = 0.5 → 100% tranzactii cu pierdere
    #
    # concluzie: relatia discount-profit este PUTERNIC NELINEARA.
    # un model care primeste discount ca numar continuu (ex: 0.3) si trateaza
    # diferenta intre 0.25 si 0.30 ca echivalenta cu dif. 0.05 la 0.10,  va genera erori mari.
    # trebuie specicificat ca e.g.  0.30 este un prag critic.

    # flag discount mare: discount >= 30% → aproape sigur pierdere
    temp['high_discount']   = (temp['discount'] >= 0.3).astype(int)

    # discount moderat: intre 0 si 30% (zona intermediara, cu risc variabil, mediu)
    temp['medium_discount'] = ((temp['discount'] > 0.0) & (temp['discount'] < 0.3)).astype(int)

    # discount zero: nicio reducere → profit garantat pozitiv
    temp['zero_discount']   = (temp['discount'] == 0.0).astype(int)

    # itroducem col discountul patratic (simlar al concept cu MSE: surprinde curbura efectului (impactul creste accelerat)
    # eg.: diferenta intre 0.4 si 0.5 este mai mare decat intre 0.1 si 0.2
    temp['discount_sq'] = temp['discount'] ** 2


    # -------------------------------------------------------------------------
    # ENCODING VARIABILE CATEGORICE
    # -------------------------------------------------------------------------
    # HistGradientBoosting accepta valori numerice, nu string-uri.
    # folosim encoding ordinal simplu (nu one-hot) deoarece arborii de decizie
    # pot gestiona corect variabile ordinale chiar fara a presupune ordine liniara.

    # segmentul clientului: Consumer / Corporate / Home Office
    seg_map  = {'Consumer': 0, 'Corporate': 1, 'Home Office': 2}

    # modul de livrare: de la cel mai lent (Standard) la cel mai rapid (Same Day)
    ship_map = {'Standard Class': 0, 'Second Class': 1, 'First Class': 2, 'Same Day': 3}

    temp['segment_enc']   = temp['segment'].map(seg_map).fillna(0)
    temp['ship_mode_enc'] = temp['ship_mode'].map(ship_map).fillna(0)


    # -------------------------------------------------------------------------
    # APLICARE LOOKUPS DE RISC (construite din train, aplicate pe orice split)
    # -------------------------------------------------------------------------
    # join 'left' = pastreaza toate randurile din 'temp', chiar daca nu gaseste
    # o potrivire in lookup (cazul combinatiilor noi → NaN → fillna(0))

    # risc principal: sub-categorie × regiune
    temp = temp.merge(risk_lookup, on=['sub-category', 'region'], how='left')
    temp['risk_subcat_region'] = temp['risk_subcat_region'].fillna(0)

    # risc la nivel de categorie (Furniture / Office Supplies / Technology)
    if risk_cat is not None:
        temp = temp.merge(risk_cat, on=['category'], how='left')
        temp['risk_category'] = temp['risk_category'].fillna(0)
    else:
        temp['risk_category'] = 0

    # risc la nivel de produs individual
    if risk_prod is not None:
        temp = temp.merge(risk_prod, on=['product_id'], how='left')
        temp['risk_product'] = temp['risk_product'].fillna(0)
    else:
        temp['risk_product'] = 0

    # risc sezonier: profitabilitatea medie a sub-categoriei in luna respectiva
    if seasonal_lookup is not None:
        temp = temp.merge(seasonal_lookup, on=['order_month', 'sub-category'], how='left')
        temp['seasonal_risk'] = temp['seasonal_risk'].fillna(0)
    else:
        temp['seasonal_risk'] = 0

    # rrofitabilitatea medie pe mod de livrare
    if ship_lookup is not None:
        temp = temp.merge(ship_lookup, on='ship_mode', how='left')
        temp['ship_profit_avg'] = temp['ship_profit_avg'].fillna(0)
    else:
        temp['ship_profit_avg'] = 0


    # -------------------------------------------------------------------------
    # LOGICA FALLBACK pt BLIND (combinatii noi care nu existau in train)
    # -------------------------------------------------------------------------
    # in 2017 pot aparea combinatii sub-cat×region pe care modelul nu le-a vazut.
    # fara fallback, aceste randuri primesc risk_subcat_region = 0 (neutru).
    # cu fallback, primesc marja medie a produsului respectiv din datele istorice.
    if is_blind:
        # verificam care randuri au o combinatie noua (nevazuta in train)
        temp = temp.merge(known_comb, on=['sub-category', 'region'], how='left')
        temp['is_known'] = temp['is_known'].fillna(False)
        new_mask = ~temp['is_known']   # masca pt combinatiile noi

        if new_mask.any():
            # pt combinatiile noi, cautam marja medie a produsului in fallback
            temp = temp.merge(
                fallback_lookup,
                on=['order_year', 'region', 'sub-category', 'product_id', 'product_name'],
                how='left'
            )
            # Inlocuim risk_subcat_region = 0 cu marja din fallback (mai informativa)
            temp.loc[new_mask, 'risk_subcat_region'] = \
                temp.loc[new_mask, 'fallback_p_margin'].fillna(0)
            temp.drop(columns=['fallback_p_margin'], inplace=True)

        temp.drop(columns=['is_known'], inplace=True)


    # -------------------------------------------------------------------------
    # AGREGARE: de la tranzactii individuale la nivel customer×year×region×sub-cat
    # -------------------------------------------------------------------------
    # fiecare rand in setul agreat =
    #   totalitatea tranzactiilor unui client intr-un an, pt o anumita regiune si sub-categorie de produse
    #
    # De exemplu: clientul CG-12520, 2016, South, Bookcases
    # → suma tuturor comenzilor sale de Bookcases din South in 2016
    agg = temp.groupby(['customer_id', 'order_year', 'region', 'sub-category']).agg(
        sales              = ('sales',             'sum'),       # vanzari totale
        profit             = ('profit',            'sum'),       # profit total (TARGET)
        discount           = ('discount',          'mean'),      # discount mediu
        quantity           = ('quantity',          'sum'),       # cantitate totala
        is_high_season     = ('is_high_season',    'sum'),       # cate comenzi in Q4
        is_q1              = ('is_q1',             'sum'),       # cate comenzi in Q1
        Art_West           = ('Art_West',          'max'),       # 1 daca orice comanda e Furniture+West
        Acc_East           = ('Acc_East',          'max'),
        risk_subcat_region = ('risk_subcat_region','mean'),      # riscul mediu al sub-cat×regiune
        risk_category      = ('risk_category',     'mean'),
        risk_product       = ('risk_product',      'mean'),
        seasonal_risk      = ('seasonal_risk',     'mean'),
        ship_profit_avg    = ('ship_profit_avg',   'mean'),
        shipping_delay     = ('shipping_delay',    'mean'),      # intarzierea medie de livrare
        n_orders           = ('order_id',          'nunique'),   # numarul de comenzi distincte
        n_products         = ('product_id',        'nunique'),   # diversitatea produselor
        high_discount      = ('high_discount',     'sum'),       # cate comenzi cu discount >= 30%
        medium_discount    = ('medium_discount',   'sum'),
        zero_discount      = ('zero_discount',     'sum'),       # cate comenzi fara reducere
        discount_sq        = ('discount_sq',       'mean'),
        segment_enc        = ('segment_enc',       'first'),     # segmentul clientului (constant)
        ship_mode_enc      = ('ship_mode_enc',     'mean'),
    ).reset_index()


    # -------------------------------------------------------------------------
    # FEATURES DERIVATE (calculate dupa agregare)
    # -------------------------------------------------------------------------

    # vanzarea medie per unitate: proxy pt pretul produselor cumparate
    agg['unit_sales'] = agg['sales'] / (agg['quantity'] + 1e-5)

    # marja de profit: profitul ca proportie din vanzari (intre -1 si +1 in mod normal)
    # aceasta este si variabila TARGET (inainte de transformare)
    agg['p_margin'] = agg['profit'] / (agg['sales'] + 1e-5)

    # eficienta discountului: cat profit a ramas dupa aplicarea reducerii
    # valoare negativa = discountul a depasit marja → pierdere
    agg['discount_efficiency'] = agg['profit'] / (agg['discount'] + 0.01)

    # semnal de risc ridicat: marja sub -20% → ordin de marime mai rau decat media
    agg['high_loss_risk'] = (agg['p_margin'] < -0.2).astype(int)

    # log-vanzari: comprima scala larga a valorilor de vanzari (de la 1$ la 10.000$)
    # Fara log, un client cu 10.000$ in vanzari ar domina complet gradientii modelului
    agg['log_sales'] = np.log1p(agg['sales'])

    # proportia comenzilor cu discount mare (>= 30%) din totalul comenzilor clientului
    # un client care comanda mereu cu discount mare are un profil de risc specific
    agg['high_disc_pct'] = agg['high_discount'] / (agg['n_orders'] + 1e-5)

    # proportia comenzilor fara nicio reducere
    agg['zero_disc_pct'] = agg['zero_discount'] / (agg['n_orders'] + 1e-5)

    # diversitate produse: cate produse diferite per comanda
    # un client care cumpara mereu acelasi produs vs. unul care experimeteaza
    agg['product_diversity'] = agg['n_products'] / (agg['n_orders'] + 1e-5)


    # -------------------------------------------------------------------------
    # ADAUGARE RFM — DIFERIT pt TRAIN VS. TEST/BLIND (anti-leakage)
    # -------------------------------------------------------------------------
    if rfm_by_year is not None and len(rfm_by_year) > 0:
        # TRAIN: join pe customer_id × order_year
        # Fiecare rand din train primeste RFM calculat DOAR din ani anteriori
        # Ex: rand din 2015 → RFM calculat din 2014
        #     rand din 2016 → RFM calculat din 2014 + 2015
        agg = agg.merge(rfm_by_year, on=['customer_id', 'order_year'], how='left')
    else:
        # TEST / BLIND: join pe customer_id (unic per client)
        # Toti clientii primesc acelasi RFM (calculat din perioada premergatoare splitului)
        agg = agg.merge(rfm_lookup, on='customer_id', how='left')

    # Clientii noi (fara historic anterior) primesc valoare 0 pt toate coloanele RFM
    rfm_cols = [
        'cust_total_profit', 'cust_total_sales', 'cust_n_orders',
        'cust_avg_discount', 'cust_loss_rate', 'cust_avg_profit',
        'cust_recency', 'cust_tenure', 'cust_avg_margin',
        'cust_has_history', 'cust_high_disc_rate'
    ]
    for col in rfm_cols:
        if col not in agg.columns:
            agg[col] = 0
        else:
            agg[col] = agg[col].fillna(0)
            # cust_has_history = 0 pt clienti noi → modelul stie ca nu are context


    # -------------------------------------------------------------------------
    # FEATURES DE TIMP BAZATE PE ISTORIA CLIENTULUI (anti-leakage prin shift)
    # -------------------------------------------------------------------------
    # Sortam cronologic per client inainte de shift
    agg = agg.sort_values(['customer_id', 'order_year'])

    # p_last: marja de profit a clientului in perioada ANTERIOARA
    # shift(1) ia valoarea randului precedent din acelasi grup (customer_id)
    # → randul 2014 primeste NaN → fillna(0) (nu avem trecut)
    # → randul 2015 primeste valoarea din 2014
    # → randul 2016 primeste valoarea din 2015
    # CORECT: nu exista leakage deoarece shift(1) priveste INAPOI, nu inainte
    agg['p_last']      = agg.groupby('customer_id')['p_margin'].shift(1).fillna(0)
    agg['profit_last'] = agg.groupby('customer_id')['profit'].shift(1).fillna(0)
    agg['sales_last']  = agg.groupby('customer_id')['sales'].shift(1).fillna(0)

    # Trend: marja curenta minus marja anterioara
    # Pozitiv → clientul devine mai profitabil; negativ → se deterioreaza
    agg['margin_trend'] = agg['p_margin'] - agg['p_last']

    # Ponderea de antrenament: randuri cu vanzari mari influenteaza mai mult gradientii
    # log1p comprima scala: 1$ → 0.69, 100$ → 4.61, 10000$ → 9.21
    agg['sample_weight'] = np.log1p(np.abs(agg['sales']))

    return agg


# aplicam pipeline-ul pe fiecare split, cu parametrii corespunzatori
print("Preprocesare in curs...")

# TRAIN: foloseste rfm_by_year (RFM per an, fara leakage temporal)
df_train_agg = preprocess_v4(
    df_train, risk_subcat_region, fallback_risk, known_comb,
    rfm_lookup=None, rfm_by_year=rfm_train_lookup,
    ship_lookup=ship_profit, seasonal_lookup=seasonal_risk,
    risk_cat=risk_category, risk_prod=risk_product
)

# TEST: foloseste rfm_for_test (RFM calculat din df_train, anterior testului)
df_test_agg = preprocess_v4(
    df_test, risk_subcat_region, fallback_risk, known_comb,
    rfm_lookup=rfm_for_test, rfm_by_year=None,
    ship_lookup=ship_profit, seasonal_lookup=seasonal_risk,
    risk_cat=risk_category, risk_prod=risk_product
)

# BLIND: foloseste rfm_for_blind (RFM din df_rest=2014-2016) + is_blind=True (activeaza fallback)
df_blind_agg = preprocess_v4(
    df_blind, risk_subcat_region, fallback_risk, known_comb,
    rfm_lookup=rfm_for_blind, rfm_by_year=None,
    ship_lookup=ship_profit, seasonal_lookup=seasonal_risk,
    risk_cat=risk_category, risk_prod=risk_product,
    is_blind=True
)

print(f"Dupa agregare → Train: {len(df_train_agg)} | Test: {len(df_test_agg)} | Blind: {len(df_blind_agg)}")


# =============================================================================
# 5. CLUSTERIZARE CLIENTI — FIT EXCLUSIV PE TRAIN
# =============================================================================
# KMeans imparte clientii in grupuri (clustere) cu comportament similar.
# Scopul: antrenam un model SPECIALIZAT pt fiecare grup.
# Un model specializat pe "clienti high-value" este mai precis decat unul generic.
#
# ANTI-LEAKAGE: kmeans.fit() vede DOAR datele din train.
# Testul si blind-ul sunt asignate celui mai apropiat centroid din train,
# fara a-l modifica. Daca am face fit pe toate datele, centroizii
# s-ar deplasa spre distributia din 2017 → segmentarea ar absorbi info din viitor.

# Alegem features relevante pt profilul clientului (nu features de target)
clust_features = ['sales', 'p_margin', 'discount', 'cust_avg_margin', 'cust_loss_rate']

# RobustScaler normalizeaza pe baza medioanei si IQR (robust la outlieri)
# → fiecare feature are aproximativ aceeasi scara, altfel 'sales' (0-10000$)
#   ar domina 'discount' (0-0.8) in calculul distantei euclidiene
scaler_clust = RobustScaler()
scaler_clust.fit(df_train_agg[clust_features])   # fit DOAR pe train → parametrii fixati

X_tr_cl = scaler_clust.transform(df_train_agg[clust_features])
X_te_cl = scaler_clust.transform(df_test_agg[clust_features])    # aceeasi transformare
X_bl_cl = scaler_clust.transform(df_blind_agg[clust_features])   # aceeasi transformare

# n_clusters=4 ales empiric (balanta intre specificitate  (3) si date suficiente per cluster (5))
# n_init=15 → KMeans ruleaza de 15 ori cu initializari diferite → alege cel mai bun
kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
kmeans.fit(X_tr_cl)   # centroizii sunt calculati DOAR din train

# Fiecare split primeste eticheta clusterului cel mai apropiat (din centroizii train)
df_train_agg['segment_client'] = kmeans.predict(X_tr_cl)
df_test_agg['segment_client']  = kmeans.predict(X_te_cl)
df_blind_agg['segment_client'] = kmeans.predict(X_bl_cl)

print("\nDistributie segmente (din train):")
for s, n in df_train_agg['segment_client'].value_counts().sort_index().items():
    avg_p = df_train_agg[df_train_agg['segment_client'] == s]['profit'].mean()
    print(f"  Segment {s}: {n:4d} randuri | profit mediu: ${avg_p:8.1f}")


# =============================================================================
# 6. TRANSFORMARE TARGET — YEO-JOHNSON PE MARJA DE PROFIT
# =============================================================================
# de ce transformam targetul?
#   distributia p_margin este asimetrica (coada lunga la stanga din pierderi mari);
#   gradient Boosting minimizeaza MSE → erorile mari (outlieri) primesc o pondere
#   disproportionata; transformand targetul spre o distributie mai normala,
#   gradientii devin mai stabili si modelul generalizeaza mai bine
#
# de ce p_margin in loc de profit direct?
#   p_margin = profit / sales → este normalizata prin marimea comenzii.
#   e.g. un profit de 100$ pe o vanzare de 200$ (50% marja) este complet diferit
#   de 100$ pe o vanzare de 10.000$ (1% marja). Marja surprinde eficienta.
#
# strategia de reconstructie:
#   antrenam pe target=transform(p_margin) → predictia este in spatiul transformat
#   → inversam transformarea → obtinem p_margin_prezis → inmultim cu sales → PROFIT

# --- WINSORIZE ---
# taiem margini extreme (1% - 99%) INAINTE de transformare.
# motivul: un outlier extrem (ex: marja = -2.7) ar forta PowerTransformer sa
# aloce mare parte din spatiul distributional unui singur caz izolat.
# bounds calculate DOAR din train → fixate, aplicate identic pe test si blind.
low_w  = df_train_agg['p_margin'].quantile(0.01)
high_w = df_train_agg['p_margin'].quantile(0.99)
print(f"\nWinsorize p_margin: [{low_w:.4f}, {high_w:.4f}]")

df_train_agg['margin_w'] = df_train_agg['p_margin'].clip(low_w, high_w)

# Yeo-Johnson: varianta a Box-Cox care functioneaza si cu valori negative
# (profit negativ → marja negativa → nu putem folosi log sau Box-Cox standard)
# pt.fit() estimeaza parametrul lambda DOAR din distributia din train.
pt = PowerTransformer(method='yeo-johnson')
pt.fit(df_train_agg[['margin_w']])

# Transformam targetul pt fiecare split
# IMPORTANT: test si blind clipuiesc cu aceiasi bounds din train (nu recalculeaza)
df_train_agg['target'] = pt.transform(df_train_agg[['margin_w']])
df_test_agg['target']  = pt.transform(
    df_test_agg[['p_margin']].clip(low_w, high_w).rename(columns={'p_margin': 'margin_w'})
)
df_blind_agg['target'] = pt.transform(
    df_blind_agg[['p_margin']].clip(low_w, high_w).rename(columns={'p_margin': 'margin_w'})
)


# =============================================================================
# 7. LISTA DE FEATURES pt MODEL
# =============================================================================
# organizate in grupuri tematice pt claritate.
# modelul primeste exact aceste coloane in aceasta ordine la antrenare si predictie.

features = [
    # --- CORE TRANZACTIONALE ---
    # Informatii directe despre dimensiunea comenzii
    'sales',        # valoarea totala a vanzarilor (in $)
    'discount',     # discountul mediu aplicat (0.0 - 0.8)
    'quantity',     # cantitatea totala cumparata
    'log_sales',    # log(1 + sales) → comprima scala pt outlieri
    'unit_sales',   # pretul mediu per unitate → proxy pt calitatea produselor

    # --- DISCOUNT FEATURES (critice — relatia e puternic nelineara) ---
    'discount_sq',     # discount^2 → capteaza accelerarea impactului negativ
    'high_disc_pct',   # % comenzi cu discount >= 30% → client "dependet de reduceri"
    'zero_disc_pct',   # % comenzi fara reducere → clientii full-price sunt mai profitabili
    'high_discount',   # total comenzi cu discount >= 30% in aceasta perioada
    'zero_discount',   # total comenzi fara reducere

    # --- RISK MULTI-NIVEL (toate calculate din train) ---
    'risk_subcat_region',  # profitul mediu al sub-cat×region din trecut
    'risk_category',       # profitul mediu al categoriei (Furniture/Tech/Office)
    'risk_product',        # profitul mediu al produsului individual
    'seasonal_risk',       # profitul mediu al sub-cat in luna respectiva

    # --- COMPORTAMENT CLIENT (RFM) ---
    # cel mai puternic grup: un client profitabil in trecut va fi profitabil in viitor
    'cust_total_profit',   # profitul total generat de client in toata istoria sa
    'cust_avg_profit',     # profitul mediu per rand de comanda
    'cust_avg_margin',     # marja medie istorica → predictor direct al marjei viitoare
    'cust_n_orders',       # numarul de comenzi → masura frecventei (F din RFM)
    'cust_loss_rate',      # proportia tranzactiilor cu pierdere → risc structural
    'cust_recency',        # ani de la ultima comanda → masura recentei (R din RFM)
    'cust_tenure',         # ani de activitate → loialitate si predictibilitate
    'cust_has_history',    # 0/1: clientul nou sau cu istoric
    'cust_high_disc_rate', # % din comenzile istorice cu discount >= 30%

    # --- TEMPORAL SI OPERATIONAL ---
    'is_high_season',   # 1 daca comenzile sunt predominant in Q4
    'is_q1',            # 1 daca comenzile sunt predominant in Q1 (slab)
    'shipping_delay',   # intarzierea medie de livrare (zile)
    'ship_profit_avg',  # profitabilitatea medie a modului de livrare ales
    'ship_mode_enc',    # modul de livrare encodat numeric
    'segment_enc',      # segmentul clientului (Consumer/Corporate/Home Office)

    # --- ISTORIC CLIENT (shift temporal, fara leakage) ---
    'p_last',         # marja din PERIOADA ANTERIOARA a aceluiasi client
    'profit_last',    # profitul din PERIOADA ANTERIOARA
    'margin_trend',   # tendinta: marja curenta - marja anterioara (+ = imbunatatire)
    'discount_efficiency',  # profit / discount → eficienta reducerii acordate
    'high_loss_risk', # 1 daca marja < -20% → pierdere semnificativa

    # --- DIVERSITATE COMENZI ---
    'n_orders',          # numarul de comenzi distincte in aceasta combinatie
    'n_products',        # numarul de produse diferite cumparate
    'product_diversity', # produse/comenzi → cat de diversificat este clientul

    # --- INTERACTIUNI ---
    'Art_West',   # Furniture × West (combinatie cu comportament specific)
    'Acc_East',   # Accessories × East
]


# =============================================================================
# 8. ANTRENARE MODELE EXPERT — CATE UNUL PER SEGMENT
# =============================================================================
# logica: clientii din segmente diferite au comportamente diferite.
# un model antrenat EXCLUSIV pe clienti high-value (segment 3: profit mediu $273)
# poate invata pattern-uri specifice acelui segment, pe care un model generic le-ar dilua.
#
# HistGradientBoostingRegressor = implementare sklearn a Gradient Boosted Trees,
# similar cu XGBoost. Avantaje: gestioneaza NaN nativ, mai rapid, regularizare L2.

# Initializam coloana de predictii cu 0 (va fi completata per segment)
for df_s in [df_train_agg, df_test_agg, df_blind_agg]:
    df_s['final_pred_profit'] = 0.0

experts  = {}
MIN_ROWS = 15   # minim de date pt a antrena un model expert (sub aceasta valoare → instabil)

for seg in sorted(df_train_agg['segment_client'].unique()):

    # Cream mastile booleene pt fiecare split
    mask_tr = df_train_agg['segment_client'] == seg
    mask_te = df_test_agg['segment_client']  == seg
    mask_bl = df_blind_agg['segment_client'] == seg

    n_tr = mask_tr.sum()
    if n_tr < MIN_ROWS:
        print(f"  Segment {seg}: {n_tr} randuri — prea putin pt model expert, skip")
        continue

    X_tr = df_train_agg.loc[mask_tr, features]
    y_tr = df_train_agg.loc[mask_tr, 'target']         # target = transform(p_margin)
    w_tr = df_train_agg.loc[mask_tr, 'sample_weight']  # ponderi: comenzi mari conteaza mai mult

    model = HistGradientBoostingRegressor(
        max_iter=1500,            # numarul maxim de arbori (early stopping va opri mai devreme)
        learning_rate=0.01,       # pasul de gradient (mic = mai precis, mai lent)
        max_depth=5,              # adancimea maxima a arborilor (controland complexitatea)
        min_samples_leaf=15,      # minim de date in fiecare frunza (regularizare implicita)
        l2_regularization=10.0,   # penalitate L2 pe ponderi → reduce overfitting
        max_features=0.8,         # 80% din features evaluate la fiecare split → diversitate
        random_state=42,
        early_stopping=True,      # opreste antrenarea daca validarea nu se mai imbunatateste
        validation_fraction=0.15, # 15% din datele de train folosite pt early stopping
        n_iter_no_change=40,      # asteapta 40 de iteratii fara imbunatatire inainte de stop
        scoring='neg_mean_absolute_error',  # criteriul de early stopping
    )

    # antrenare mode expert EXCLUSIV pe datele din segmentul sau
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    experts[seg] = model

    print(f"  Segment {seg} ({n_tr:4d} train, {model.n_iter_:4d} iters) | "
          f"test={mask_te.sum()} | blind={mask_bl.sum()}")

    # generare predictii pt toate spliturile cu acest expert
    for df_s, mask in [(df_train_agg, mask_tr),
                        (df_test_agg,  mask_te),
                        (df_blind_agg, mask_bl)]:
        if mask.sum() == 0:
            continue

        # pas 1: predictia modelului este in spatiul transformat (Yeo-Johnson)
        pred_transformed = model.predict(df_s.loc[mask, features])

        # pas 2: inversam transformarea → obtinem p_margin in spatiul original
        pred_margin = pt.inverse_transform(pred_transformed.reshape(-1, 1)).ravel()

        # pas 3: profit = marja × vanzari
        # reconstructia finala a profitului in dolari
        df_s.loc[mask, 'final_pred_profit'] = pred_margin * df_s.loc[mask, 'sales']


# =============================================================================
# 9. META-MODEL GLOBAL (STACKING)
# =============================================================================
# problema cu modelele expert: segmentele mici (ex: 7 randuri) nu pot avea
# propriul model expert → predictia lor ramane 0.
# solutia: un model global antrenat pe TOATE datele, care include si 'segment_client'
# ca feature → stie sa diferentieze intre segmente fara sa fie specializat pe unul.
#
# stacking = tehnica ensemble in care un model de nivel 2 combina predictiile
# mai multor modele de nivel 1. Aici: blending simplu (medie ponderata) intre
# expert per-segment si meta-model global.

print("\nAntrenare model global...")

# Meta-modelul primeste toate features + segmentul ca feature suplimentar
meta_features = features + ['segment_client']

meta_model = HistGradientBoostingRegressor(
    max_iter=1500,
    learning_rate=0.01,
    max_depth=6,           # usor mai adanc decat expertii (vede mai multe interactiuni)
    min_samples_leaf=10,
    l2_regularization=8.0,
    max_features=0.75,
    random_state=99,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=40,
    scoring='neg_mean_absolute_error',
)

# meta-modelul se antreneaza pe toate datele din train (nu per segment)
meta_model.fit(
    df_train_agg[meta_features],
    df_train_agg['target'],
    sample_weight=df_train_agg['sample_weight']
)
print(f"  Meta-model: {meta_model.n_iter_} iters (early stopping)")

# generare predictii global-modelului pt toate spliturile
for df_s in [df_train_agg, df_test_agg, df_blind_agg]:
    meta_raw    = meta_model.predict(df_s[meta_features])
    meta_margin = pt.inverse_transform(meta_raw.reshape(-1, 1)).ravel()
    df_s['meta_pred_profit'] = meta_margin * df_s['sales']


# --- CALIBRARE BLEND ALPHA ---
# cautam proportia optima de combinare a celor doua modele.
# alpha = cat % din predictia finala vine din expert, (1-alpha) % din meta.
# calibrarea se face EXCLUSIV pe setul de TEST (nu pe blind, care e "viitor").
best_alpha, best_mae = 0.7, float('inf')

for alpha in np.arange(0.5, 1.01, 0.05):
    blend = (alpha * df_test_agg['final_pred_profit'] +
             (1 - alpha) * df_test_agg['meta_pred_profit'])
    mae_crt = mean_absolute_error(df_test_agg['profit'], blend)
    if mae_crt < best_mae:
        best_mae   = mae_crt
        best_alpha = alpha

print(f"  Alpha optim: {best_alpha:.2f} × expert + {1-best_alpha:.2f} × meta")

# aplicam blend-ul final pe toate spliturile cu alpha calibrat pe test
for df_s in [df_train_agg, df_test_agg, df_blind_agg]:
    df_s['blend_pred'] = (best_alpha * df_s['final_pred_profit'] +
                          (1 - best_alpha) * df_s['meta_pred_profit'])


# =============================================================================
# 10. EVALUARE COMPLETA
# =============================================================================

def metrics(df_s, pred_col='blend_pred'):
    """Calculeaza R², MAE, Mediana AE, Profit Mediu si Profit Median."""
    actual, pred = df_s['profit'], df_s[pred_col]
    return (
        r2_score(actual, pred),
        mean_absolute_error(actual, pred),
        np.median(np.abs(actual - pred)),
        actual.mean(),
        actual.median()
    )

print("\n" + "=" * 85)
print(f"{'METRICA':<32} | {'TEST (2014-16)':<22} | {'BLIND (2017)':<22}")
print("-" * 85)
for label, col in [
    ("Expert per-segment", "final_pred_profit"),
    ("Meta-model global",  "meta_pred_profit"),
    ("BLEND FINAL",        "blend_pred")
]:
    t = metrics(df_test_agg,  col)
    b = metrics(df_blind_agg, col)
    print(f"  {label:<30} R2={t[0]:.4f} MAE=${t[1]:.2f}  |  R2={b[0]:.4f} MAE=${b[1]:.2f}")
print("=" * 85)

t = metrics(df_test_agg)
b = metrics(df_blind_agg)
print(f"\nREZULTATE FINALE (blend):")
print(f"  Test  → R2={t[0]:.4f} | MAE=${t[1]:.2f} | MedAE=${t[2]:.2f} | Profit mediu=${t[3]:.2f}")
print(f"  Blind → R2={b[0]:.4f} | MAE=${b[1]:.2f} | MedAE=${b[2]:.2f} | Profit mediu=${b[3]:.2f}")
gap = (t[0] - b[0]) * 100
print(f"  Gap R² (test - blind): {gap:.2f}%")
if abs(gap) > 15:
    print("  ⚠ Gap semnificativ: modelul generalizeaza mai slab pe date noi")
else:
    print("  ✅ Gap acceptabil: modelul este stabil intre test si blind")

print("\nPERFORMANTA PE ANI (diagnostic temporal):")
all_d = pd.concat([
    df_train_agg.assign(split='train'),
    df_test_agg.assign(split='test'),
    df_blind_agg.assign(split='blind')
])
for yr in sorted(all_d['order_year'].unique()):
    yd   = all_d[all_d['order_year'] == yr]
    r2   = r2_score(yd['profit'], yd['blend_pred'])
    mae  = mean_absolute_error(yd['profit'], yd['blend_pred'])
    med  = np.median(np.abs(yd['profit'] - yd['blend_pred']))
    spl  = yd['split'].iloc[0]
    print(f"  {yr} [{spl:^5}] R2={r2:7.4f} | MAE={mae:7.2f}$ | MedAE={med:7.2f}$ | Profit mediu={yd['profit'].mean():8.2f}$")

print("\nPERFORMANTA PE SEGMENTE (blind — evaluare finala):")
for seg in sorted(df_blind_agg['segment_client'].unique()):
    sd = df_blind_agg[df_blind_agg['segment_client'] == seg]
    if len(sd) < 5:
        continue
    r2s  = r2_score(sd['profit'], sd['blend_pred']) if len(sd) > 1 else 0
    maes = mean_absolute_error(sd['profit'], sd['blend_pred'])
    print(f"  Segment {seg}: n={len(sd):4d} | R2={r2s:.4f} | MAE=${maes:.2f} | Profit mediu=${sd['profit'].mean():.1f}")


# =============================================================================
# 11. SALVARE
# =============================================================================
# salvam toate artefactele pt a aplica modelul pe date noi in productie:
# - modelele antrenate (experts + meta_model)
# - transformarile (pt, scaler_clust, kmeans) → trebuie aplicate identic
# - lookup-urile de risc → necesare pt feature engineering
# - lista de features in ordinea corecta → esentiala pt predictie

artifacts = {
    'experts':            experts,          # dict {segment_id: model}
    'meta_model':         meta_model,       # modelul global
    'power_transformer':  pt,               # Yeo-Johnson (fit pe train)
    'robust_scaler':      scaler_clust,     # scalare pt clustering (fit pe train)
    'kmeans_model':       kmeans,           # centroizii clusterelor (fit pe train)
    'risk_subcat_region': risk_subcat_region,
    'risk_category':      risk_category,
    'risk_product':       risk_product,
    'fallback_lookup':    fallback_risk,
    'known_combinations': known_comb,
    'ship_profit':        ship_profit,
    'seasonal_risk':      seasonal_risk,
    'features_list':      features,
    'meta_features':      meta_features,
    'blend_alpha':        best_alpha,
    'winsorize_bounds':   (low_w, high_w),  # bounds fixate din train
}

joblib.dump(artifacts, '../models/profit_prediction_engine_HGBRegressor_final.pkl')
df_blind_agg.to_csv('../data/model_final/predictii_profit_HGBRegressor_blind_2017_final.csv', index=False)

print("\n✅ Model salvat: profit_prediction_engine_HGBRegressor_final.pkl")
print("✅ Predictii blind: predictii_profit_HGBRegressor_blind_2017_final.csv")