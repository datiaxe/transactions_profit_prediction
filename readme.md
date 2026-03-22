# Proiect: Profit_Prediction_Engine

➡️ Problema:    Predictia profitului este dificila din cauza variatiilor de discount si sezonalitate.
➡️ Solutia:     Un pipeline de date ‘Zero-Leakage’ care separa strict istoricul de viitor.
➡️ Obiectiv:    Modelare Machine Learning pentru prezicerea profitul viitor al unei comenzi, folosind:	-  istoricul clientului
	                - tipul produsului
	                - discount
	                - regiune
	                - sezonalitate
	                - comportament client

## 1. Structura Proiectului

├── data/                                               # Datasets    
│   ├── raw_data/                                           # Datele brute, neprocesate
│   ├── processed/                                          # Datele dupa curatare si feature engineering
│   ├── data_output/                                        # Datele de output pentru comparatie, obtinute in urma incercarilor intermediare cu alte modele
│   └── model_final/                                        # Datele de output obtinute pe setul blind, dupa rularea modelului final   
├── notebooks/                                          # Jupyter Notebooks pentru EDA si experimente
├── reports/                                            # Prezentarea proiectului in format .ppt
├── src/                                                # Antrenare modele si feature engineering
│   ├── model_XGBoost_Regressor.py                          # Feature engineering model XGBoost
│   ├── model_Sequential.py                                 # Feature engineering model Sequential
│   ├── model_RandomForestRegressor.py                      # Feature engineering model RandomForerst
│   ├── MODEL_FINAL_HistGradientBoostingRegressor.py       	# Feature engineering, antrenare si rualre meod final HGBR
	└── simulare_gradio_model_final.py        				# Simularea cu input manual a modelulului final salvat
├── models/                                             # Fisierele modelelor salvate (Sequntial draft si HGBR final)
├── requirements.txt                                    # Lista dependintelor
└── README.md
