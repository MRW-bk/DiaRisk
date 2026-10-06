import dash
from dash import dcc, html, Input, Output, State, callback_context, clientside_callback, ClientsideFunction
import dash_bootstrap_components as dbc
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
import plotly.graph_objects as go
import logging
import os
import io
import base64

# Configure logging early
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Import CatBoost ---
try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError: # pragma: no cover
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost library not found. Model loading may fail if using CatBoost. Install with: pip install catboost")
except Exception as e: # pragma: no cover
    logger.error(f"Error importing CatBoost: {e}", exc_info=True)
    CATBOOST_AVAILABLE = False
# -----------------------

# --- PDF Generation ---
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.utils import simpleSplit
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab library not found. PDF generation feature will be disabled. Install with: pip install reportlab")
# ---------------------

# --- PDF Generation Enhancements for Arabic ---

ARABIC_PDF_FONT_NAME = "ArabicReportFont"
ARABIC_FONT_FILENAME = "Amiri-Regular.ttf"

# Adjust path check for assets folder
arabic_font_path_candidate = os.path.join('assets', ARABIC_FONT_FILENAME)
# Check if the font file exists in the assets folder
if os.path.exists(arabic_font_path_candidate):
    ARABIC_FONT_PATH_FULL = arabic_font_path_candidate
else:
    # If not in assets, check current directory as a fallback (less recommended for deployment)
    if os.path.exists(ARABIC_FONT_FILENAME): # pragma: no cover
         ARABIC_FONT_PATH_FULL = ARABIC_FONT_FILENAME
    else:
         ARABIC_FONT_PATH_FULL = None # pragma: no cover


REPORTLAB_ARABIC_TOOLS_AVAILABLE = False
_arabic_reshaper_module = None
_bidi_get_display_func = None

def initialize_arabic_pdf_support():
    global REPORTLAB_ARABIC_TOOLS_AVAILABLE, _arabic_reshaper_module, _bidi_get_display_func
    if not REPORTLAB_AVAILABLE: # pragma: no cover
        return

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import arabic_reshaper
        from bidi.algorithm import get_display as bidi_get_display

        _arabic_reshaper_module = arabic_reshaper
        _bidi_get_display_func = bidi_get_display

        if ARABIC_FONT_PATH_FULL and os.path.exists(ARABIC_FONT_PATH_FULL):
            pdfmetrics.registerFont(TTFont(ARABIC_PDF_FONT_NAME, ARABIC_FONT_PATH_FULL))
            logger.info(f"Successfully registered Arabic font '{ARABIC_PDF_FONT_NAME}' from '{ARABIC_FONT_PATH_FULL}' for PDF generation.")
            REPORTLAB_ARABIC_TOOLS_AVAILABLE = True
        else: # pragma: no cover
            logger.warning(
                f"Arabic font '{ARABIC_FONT_FILENAME}' not found. Tried path: '{arabic_font_path_candidate}'. "
                f"Arabic text in PDF will NOT render correctly. Please place '{ARABIC_FONT_FILENAME}' in the 'assets' folder."
            )
            REPORTLAB_ARABIC_TOOLS_AVAILABLE = False

    except ImportError: # pragma: no cover
        logger.warning(
            "Missing libraries for full Arabic PDF support (arabic_reshaper, python-bidi). "
            "Install with: pip install arabic_reshaper python-bidi. Arabic PDF may not render correctly."
        )
        REPORTLAB_ARABIC_TOOLS_AVAILABLE = False
    except Exception as e: # pragma: no cover
        logger.error(f"Error during Arabic PDF support initialization: {e}", exc_info=True)
        REPORTLAB_ARABIC_TOOLS_AVAILABLE = False

if REPORTLAB_AVAILABLE:
    initialize_arabic_pdf_support()
# -------------------------------------------


def calculate_age(birth_date_str):
    if not birth_date_str:
        return None
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d").date()
        today = date.today()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        if 0 <= age <= 120:
            return age
        else:
            # Handle potentially invalid or future dates resulting in out-of-range age
            logger.warning(f"Calculated age {age} from DOB {birth_date_str} is out of valid range (0-120).") # pragma: no cover
            return None # pragma: no cover
    except ValueError:
        # Handle invalid date formats
        logger.warning(f"Invalid date format for DOB: {birth_date_str}. Could not calculate age.") # pragma: no cover
        return None # pragma: no cover
    except Exception as e:
         # Catch any other unexpected errors during calculation
         logger.error(f"Unexpected error calculating age from DOB {birth_date_str}: {e}", exc_info=True) # pragma: no cover
         return None # pragma: no cover


# --- Updated Model File Names ---
MODEL_FILE = 'catboost_model.pkl'
SCALER_FILE = 'scaler_catboost.pkl'
PREDICTIONS_CSV = 'predictions_log.csv'
SIDEBAR_WIDTH_EXPANDED = "16rem"
SIDEBAR_WIDTH_COLLAPSED = "4.5rem"
light_theme_url = dbc.themes.BOOTSTRAP
dark_theme_url = dbc.themes.DARKLY

model = None
scaler = None
model_load_error = None
try:
    if not os.path.exists(MODEL_FILE): raise FileNotFoundError(f"Model file not found: '{MODEL_FILE}'") # pragma: no cover
    if not os.path.exists(SCALER_FILE): raise FileNotFoundError(f"Scaler file not found: '{SCALER_FILE}'") # pragma: no cover
    # Check if CatBoost is imported before loading a CatBoost model
    if not CATBOOST_AVAILABLE: # pragma: no cover
         raise ImportError("CatBoost library is required but not installed.")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    logger.info("CatBoost model and scaler loaded successfully.")
except FileNotFoundError as fnf_error: # pragma: no cover
    model_load_error = str(fnf_error)
    logger.error(f"Failed to load model or scaler: {model_load_error}")
except ImportError as imp_error: # pragma: no cover
     model_load_error = f"Missing library: {imp_error}. Please install required dependencies (e.g., 'pip install catboost')."
     logger.error(model_load_error)
except Exception as e: # pragma: no cover
    model_load_error = f"An unexpected error occurred loading model/scaler: {type(e).__name__} - {e}"
    logger.error(model_load_error, exc_info=True)


# --- Define the expected 12 features and placeholder means ---


FEATURE_ORDER_FOR_SCALER = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age', 'Glucose_BMI_Ratio', 'Age_BMI',
    'Insulin_Glucose_Ratio', 'Pregnancies_Age_Ratio'
]

PLACEHOLDER_MEANS = {
    'Pregnancies': 0,  # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
    'Glucose': 120.89453125,             # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
    'BloodPressure': 69.10546875,         # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
    'SkinThickness': 20.53645833,         # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
    'Insulin': 79.79947917,             # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
    'BMI': 31.99257812,                 # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
    'DiabetesPedigreeFunction': 0.4718764749303975, # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
    'Age': 33.24583333                  # Example mean - REPLACE WITH ACTUAL TRAINING MEAN
}

# CLINICAL_RANGES 
CLINICAL_RANGES = {
    "Glucose": {
        "unit": "mg/dL",
        "normal_max": 139,
        "prediabetes_min": 140,
        "prediabetes_max": 199,
        "diabetes_threshold": 200,
        "graph_ref_min": 50, # Example lower bound for graph scale
        "graph_ref_max": 139 # Corresponds to normal_max
    },
    "BloodPressure": {
        "unit": "mmHg",
        "hypertension_threshold": 90,
        "hypotension_threshold": 60,
        "graph_ref_min": 60, # Corresponds to hypotension_threshold
        "graph_ref_max": 89 # Corresponds to pre-hypertension/normal upper
    },
    "SkinThickness": {
        "unit": "mm",
        "normal_threshold": 23, # Example ref for females
        "elevated_threshold": 23, # Example ref for females
        "graph_ref_min": 5, # Example lower bound for graph scale
        "graph_ref_max": 23 # Corresponds to normal_threshold
    },
    "Insulin": {
        "unit": "mUI/L",
        "min": 16, # Example fasting/2hr post-glucose range
        "max": 166, # Example fasting/2hr post-glucose range
        "graph_ref_min": 16, # Corresponds to min
        "graph_ref_max": 166 # Corresponds to max
    },
    "BMI": {
        "unit": "kg/m²",
        "underweight_threshold": 18.5,
        "normal_min": 18.5,
        "normal_max": 24.9,
        "overweight_min": 25,
        "overweight_max": 29.9,
        "obesity_threshold": 30,
        "graph_ref_min": 18.5, # Corresponds to normal_min
        "graph_ref_max": 24.9 # Corresponds to normal_max
    },
    "Age": {
        "unit": "years", 
        "young_adult_max": 35, # Example threshold for "younger" for clinical considerations
        "advanced_age_threshold": 65, # Example threshold for "advanced" for clinical considerations
        "graph_ref_min": 20,  # Visual reference bar for "typical adult"
        "graph_ref_max": 65,  # Visual reference bar for "typical adult"
        "hover_ref_min": 20, # For hover consistency if showing a range for Age
        "hover_ref_max": 65  # For hover consistency
    }
}


# --- Updated Translations for better French and Arabic ---
TRANSLATIONS = {
    'en': {
        'first_name_label': "First Name",
        'last_name_label': "Last Name",
        'dob_label': "Date of Birth",
        'first_name_placeholder': "Enter patient's first name",
        'last_name_placeholder': "Enter patient's last name",
        'dob_placeholder': "Select date of birth",
        'pdf_first_name_header': "First Name",
        'pdf_last_name_header': "Last Name",
        'pdf_dob_header': "Date of Birth",
        'history_header_first_name': "First Name",
        'history_header_last_name': "Last Name",
        'history_header_dob': "Date of Birth",
        'save_pdf_button_text': "Save Report (PDF)",
        'pdf_report_title': "DiaRisk - Medical Analysis Report",
        'pdf_generated_on': "Report generated on: {timestamp}",
        'pdf_patient_data_header': "Patient Input Data",
        'pdf_parameter_header': "Parameter",
        'pdf_value_header': "Value",
        'pdf_unit_header': "Unit",
        'pdf_prediction_header': "Model Prediction Result",
        'pdf_considerations_header': "Clinical Considerations Based on Inputs",
        'pdf_no_considerations': "No specific clinical considerations generated based on the provided inputs and defined thresholds.",
        'alert_pdf_generation_error': "Error generating PDF report. Please check logs.",
        'alert_pdf_disabled': "PDF generation requires the 'reportlab' library.",
        'alert_pdf_arabic_tools_missing': "PDF generation for Arabic requires additional tools (arabic_reshaper, python-bidi) and an Arabic font. Please install them.",
        'alert_pdf_no_clinical_data': "Please enter patient clinical data and run prediction first to generate a meaningful report.",
        'app_title': "DiaRisk - Diabetes Risk Assessment (Clinical)",
        'sidebar_header': " DiaRisk",
        'sidebar_home': "Home",
        'sidebar_predictor': "Risk Predictor",
        'sidebar_history': "Patient History",
        'sidebar_about_model': "About Model",
        'theme_switch_label': "Dark Mode",
        'close_button': "Close",
        'predict_button_text': "Run Prediction",
        'reset_button_text': "Reset Form",
        'save_button_text': "Save Result",
        'info_modal_title': "DiaRisk - Information",
        'info_modal_accordion_inputs_title': "Input Parameters",
        'info_modal_accordion_inputs_p1': "Input standard patient metrics. Tooltips provide context based on common clinical guidelines or thresholds used in the model training data.",
        'info_modal_inputs_glucose': "Glucose (mg/dL): 2hr post-load plasma glucose. Normal Tolerance: <{normal_max} mg/dL. Impaired Tolerance (Prediabetes): {prediabetes_min}-{prediabetes_max} mg/dL. Diabetes: ≥{diabetes_threshold} mg/dL. Example: 110",
        'info_modal_inputs_bp': "Diastolic BP (mmHg): Diastolic blood pressure. Hypertension: >{hypertension_threshold} mmHg. Hypotension: <{hypotension_threshold} mmHg. Example: 75",
        'info_modal_inputs_skinthickness': "Skin Thickness (mm): Triceps skinfold thickness. Normal (female ref.): ≤{normal_threshold} mm. Elevated: >{elevated_threshold} mm. Example: 25",
        'info_modal_inputs_insulin': "Insulin (mUI/L): 2-hour serum insulin. Normal range: {min}-{max} mUI/L. Example: 80",
        'info_modal_inputs_bmi': "BMI (kg/m²): Body Mass Index. Underweight: <{underweight_threshold}. Normal: {normal_min}-{normal_max}. Overweight: {overweight_min}-{overweight_max}. Obesity: ≥{obesity_threshold}. Example: 28.5",
        'info_modal_inputs_age': "Age (years): Patient age. Advanced age (e.g., >{advanced_age_threshold} years) is a significant non-modifiable risk factor. Example: 55. Auto-calculated if DOB provided.",
         # New Translations for Pregnancies and DPF Info Modal
        'info_modal_inputs_pregnancies': "Pregnancies: Number of times pregnant. Includes live births, stillbirths, ectopic pregnancies, etc. Higher numbers are associated with increased risk. Example: 1",
        'info_modal_inputs_dpf': "Diabetes Pedigree Function: A genetic score function which estimates risk of diabetes based on family history. Higher score indicates higher risk. Example: 0.5",
        'select_language': 'Select Language',
        'current_language': 'Current Language: {lang}',
        'legal_disclaimer': "This application does not replace medical advice. For use by qualified healthcare professionals.",
        'landing_title': "DiaRisk - Diabetes Risk Assessment",
        'landing_subtitle': "Machine learning support for clinical risk stratification.",
        'landing_button': "Open Predictor",
        'landing_error_title': "Application Error",
        'landing_error_message': "Error loading prediction components: {error}",
        'landing_error_suggestion': "Please ensure the model and scaler files exist and are accessible, or check server logs for more details.",
        'dashboard_title': "DiaRisk - Predictor",
        'card_header_input': "Patient Data Input",
        'input_label_glucose': "Glucose (mg/dL)",
        'input_tooltip_glucose': "2hr post-load. Normal: <140. Prediabetes: 140-199. Diabetes: ≥200. Example: 110",
        'input_label_bp': "Diastolic BP (mmHg)",
        'input_tooltip_bp': "Hypertension: >90. Hypotension: <60. Example: 75",
        'input_label_skinthickness': "Skin Thickness (mm)",
        'input_tooltip_skinthickness': "Triceps. Normal (female): ≤23mm. Elevated: >23mm. Example: 25",
        'input_label_insulin': "Insulin (mUI/L)",
        'input_tooltip_insulin': "2hr Serum. Normal: 16-166 mUI/L. Example: 80",
        'input_label_bmi': "BMI (kg/m²)",
        'input_tooltip_bmi': "Underweight <18.5. Normal 18.5-24.9. Overweight 25-29.9. Obesity ≥30. Example: 28.5",
        'input_label_age': "Age (years)",
        'input_tooltip_age': "Advanced age (e.g. >65) is a risk factor. Example: 55. Auto-calculated if DOB provided.",
        # New Translations for Pregnancies and DPF Input Labels/Tooltips
        'input_label_pregnancies': "Pregnancies",
        'input_tooltip_pregnancies': "Number of times pregnant. Example: 1",
        'input_label_dpf': "Diabetes Pedigree Function",
        'input_tooltip_dpf': "Genetic risk score based on family history. Higher score = higher risk. Example: 0.5",
        'card_header_output': "Model Prediction Output",
        'card_header_clinical': "Clinical Considerations Based on Inputs",
        'card_header_graph': "Input Visualization",
        'clinical_considerations_placeholder': "Analysis of input values relative to clinical thresholds will appear here post-prediction.",
        'clinical_considerations_header': "Key input values relative to clinical thresholds:",
        'clinical_considerations_none': "No specific clinical considerations generated based on the provided inputs and defined thresholds.",
        'pred_result_low_risk': "Model Prediction: Low Diabetes Risk",
        'pred_result_high_risk': "Model Prediction: Elevated Diabetes Risk",
        'pred_prob_low_risk': "Probability (Low Risk): {prob:.3f}",
        'pred_prob_high_risk': "Probability (Elevated Risk): {prob:.3f}",
        'alert_model_unavailable': "Prediction model not available.",
        'alert_missing_value': "Missing value for {name}. Please enter all values.",
        'alert_value_must_be_positive': "{name} must be greater than zero.",
        'alert_invalid_age': "Please enter a valid age (0-120).",
        'alert_low_bmi': "BMI must be 10 or greater.",
        'alert_negative_value': "{name} cannot be negative.",
        'alert_invalid_numeric': "Invalid numeric value for {name}. Please check inputs.",
        'alert_scaler_error': "Prediction error: Scaler component issue.",
        'alert_prediction_error': "An unexpected error occurred during prediction. Check logs.",
        'consideration_glucose_normal': "**{name} ({val:.1f} {unit})**: Normal glucose tolerance.",
        'consideration_glucose_prediabetes': "**{name} ({val:.1f} {unit})**: Impaired glucose tolerance (prédiabète). Increased risk of diabetes.",
        'consideration_glucose_diabetes': "**{name} ({val:.1f} {unit})**: Likely diabetes based on glucose level.",
        'consideration_bp_hypertensive': "**{name} ({val:.0f} {unit})**: Indicates hypertension, a risk factor for type 2 diabetes.",
        'consideration_bp_hypotensive': "**{name} ({val:.0f} {unit})**: Hypotension. May be associated with a more stable metabolism or other factors.",
        'consideration_bp_normal': "**{name} ({val:.0f} {unit})**: Diastolic blood pressure in normal range.",
        'consideration_skinthickness_normal_female': "**{name} ({val:.0f} {unit})**: Normal thickness (≤{threshold} mm for females), suggesting healthy body fat.",
        'consideration_skinthickness_elevated_female': "**{name} ({val:.0f} {unit})**: Elevated thickness (>{threshold} mm), marker of excessive adiposity.",
        'consideration_insulin_normal': "**{name} ({val:.0f} {unit})**: Within normal range (2hr post-glucose).",
        'consideration_insulin_high': "**{name} ({val:.0f} {unit})**: Elevated value (2hr post-glucose), may indicate insulin resistance.",
        'consideration_insulin_low': "**{name} ({val:.0f} {unit})**: Low value (2hr post-glucose), requires clinical evaluation.",
        'consideration_bmi_underweight': "**{name} ({val:.1f} {unit})**: Underweight.",
        'consideration_bmi_normal': "**{name} ({val:.1f} {unit})**: Normal weight.",
        'consideration_bmi_overweight': "**{name} ({val:.1f} {unit})**: Overweight (moderate diabetes risk).",
        'consideration_bmi_obesity': "**{name} ({val:.1f} {unit})**: Obesity (high diabetes risk).",
        'consideration_age_young': "**{name} ({val:.0f} {unit_age})**: Younger age, typically lower baseline risk for type 2 diabetes.",
        'consideration_age_adult': "**{name} ({val:.0f} {unit_age})**: Adult age.",
        'consideration_age_advanced': "**{name} ({val:.0f} {unit_age})**: Advanced age is a known risk factor for diabetes.",
        'warning_positive': "{name} must be > 0.",
        'warning_bmi_min': "{name} must be ≥ 10.",
        'warning_age_range': "Age must be between 0 and 120.",
        'warning_unusual_high': "{name} ({val:.1f}) seems unusually high. Verify.",
        'warning_invalid_format': "Invalid number format for {name}.",
        'warning_future_dob': "Date of Birth cannot be in the future.",
        'warning_invalid_date_format': "Invalid date format. Use YYYY-MM-DD.",
        'graph_title': "Patient Values vs. Clinical Reference Ranges",
        'graph_yaxis_label': "Value",
        'graph_legend_ref': "Reference Range",
        'graph_legend_patient': "Patient Value",
        'graph_hover_template': "<b>%{x}</b><br>Patient Value: %{y:.1f} %{customdata[2]}<br>Reference Range: %{customdata[0]:.1f}-{customdata[1]:.1f} %{customdata[2]}<extra></extra>",
        'graph_no_data_title': "Enter patient data for visualization",
        'graph_no_data_annotation': "No valid data to display.",
        'history_page_title': "Prediction History Log",
        'history_page_description': "This table shows previously logged prediction results.",
        'history_table_empty': "No prediction history found.",
        'history_header_timestamp': "Timestamp",
        'history_header_first_name': "First Name",
        'history_header_last_name': "Last Name",
        'history_header_dob': "Date of Birth",
        'history_header_glucose': "Glucose",
        'history_header_bp': "Diastolic BP",
        'history_header_skinthickness': "Skin Thickness",
        'history_header_insulin': "Insulin",
        'history_header_bmi': "BMI",
        'history_header_age': "Age",
         # New History Headers for Pregnancies and DPF
        'history_header_pregnancies': "Pregnancies",
        'history_header_dpf': "DPF", # Abbreviated for table header
        'history_header_outcome': "Predicted Outcome",
        'history_header_probability': "Probability",
        # --- Updated About Model Text ---
        'about_model_title': "About the Prediction Model (CatBoost)",
        # --- Updated P1 description ---
        'about_model_p1': "This tool utilizes a CatBoost model trained on the PIMA Indians Diabetes Database and data from 2000 patients at Frankfurt Hospital, Germany. It predicts the likelihood of diabetes based on input features and provides a risk probability.",
        # --- End Updated P1 description ---
        'about_model_p2': "CatBoost is a gradient boosting algorithm that uses oblivious decision trees. It is known for its high performance, robustness to hyperparameters, and native handling of categorical features. It builds trees sequentially, where each new tree corrects the errors of the previous ones, optimizing a differentiable loss function.",
        'about_model_features_title': "Input Features Used by the Model:",
        # Updated to reflect that Pregnancies and DPF are now user inputs
        'about_model_features_list': "The model uses the following input features collected directly from the user: Pregnancies, Glucose, Diastolic Blood Pressure, Skin Thickness (Triceps), Insulin (2hr Serum), BMI, Age, and Diabetes Pedigree Function. It also incorporates several internally calculated features derived from these inputs (e.g., ratios, interactions).",
        # --- End Refined List ---
        # Updated note to remove mention of using average values for Pregnancies and DPF
        'about_model_data_note': "Note: The model was trained primarily on female subjects of Pima Indian heritage and a cohort from a German hospital. Applicability to other populations may vary.",
        'about_model_performance_note_title': "Model Performance:",
        'about_model_performance_note_text': "Approximate cross-validated performance on the training datasets using the CatBoost model: Accuracy ~100%, Area Under the ROC Curve (AUC ROC) ~0.99. The AUC ROC measures the model's ability to distinguish between high and low risk patients across various decision thresholds, with values closer to 1 indicating better performance. These metrics are indicative and actual performance may differ in specific clinical scenarios.",
        'about_model_disclaimer': "This model provides a risk probability based on input features and is intended to supplement, not replace, clinical judgment made by qualified healthcare professionals.",
         # --- Keys for translating history table content ---
        'history_outcome_low': "Low Risk", # Internal key stored in CSV -> Display text
        'history_outcome_high': "Elevated Risk", # Internal key stored in CSV -> Display text
        # PDF Headers for new inputs
        'pdf_pregnancies_header': "Pregnancies",
        'pdf_dpf_header': "DPF", # Abbreviated for PDF table header
    },
    'fr': {
        'first_name_label': "Prénom",
        'last_name_label': "Nom de famille",
        'dob_label': "Date de Naissance",
        'first_name_placeholder': "Entrez le prénom du patient",
        'last_name_placeholder': "Entrez le nom de famille du patient",
        'dob_placeholder': "Sélectionnez la date de naissance",
        'pdf_first_name_header': "Prénom",
        'pdf_last_name_header': "Nom",
        'pdf_dob_header': "Date de Naissance",
        'history_header_first_name': "Prénom",
        'history_header_last_name': "Nom",
        'history_header_dob': "Date de Naissance",
        'save_pdf_button_text': "Enregistrer Rapport (PDF)",
        'pdf_report_title': "DiaRisk - Rapport d'Analyse Médicale",
        'pdf_generated_on': "Rapport généré le : {timestamp}",
        'pdf_patient_data_header': "Données d'Entrée Patient",
        'pdf_parameter_header': "Paramètre",
        'pdf_value_header': "Valeur",
        'pdf_unit_header': "Unité",
        'pdf_prediction_header': "Résultat Prédiction Modèle",
        'pdf_considerations_header': "Considérations Cliniques Basées sur les Entrées",
        'pdf_no_considerations': "Aucune considération clinique spécifique générée basée sur les entrées et seuils définis.",
        'alert_pdf_generation_error': "Erreur lors de la génération du rapport PDF. Vérifiez les logs.",
        'alert_pdf_disabled': "La génération PDF nécessite la librairie 'reportlab'.",
        'alert_pdf_arabic_tools_missing': "La génération de PDF pour l'arabe nécessite des outils supplémentaires (arabic_reshaper, python-bidi) et une police arabe. Veuillez les installer.",
        'alert_pdf_no_clinical_data': "Veuillez saisir les données cliniques du patient et lancer la prédiction avant de générer un rapport significatif.",
        'app_title': "DiaRisk - Évaluation Risque Diabète (Clinique)",
         'legal_disclaimer': "Cette application ne remplace pas un avis médical. Destinée aux professionnels de santé qualifiés.",
        'landing_title': "DiaRisk - Évaluation Risque Diabète",
        'landing_subtitle': "Support par apprentissage automatique pour la stratification du risque clinique.",
        'landing_button': "Ouvrir le Prédicteur",
        'landing_error_title': "Erreur d'Application",
        'landing_error_message': "Erreur lors du chargement des composants de prédiction : {error}",
        'landing_error_suggestion': "Veuillez vérifier que les fichiers modèle et scaler existent et sont accessibles, ou consultez les logs serveur pour plus de détails.",
        'dashboard_title': "DiaRisk - Prédicteur ",
        'card_header_input': "Saisie des Données Patient",
        'select_language': 'Choisir la langue',
        'sidebar_header': " DiaRisk",
        'sidebar_home': "Accueil",
        'sidebar_predictor': "Prédicteur de Risque",
        'sidebar_history': "Historique Patients",
        'sidebar_about_model': "À propos du Modèle",
        'theme_switch_label': "Mode Sombre",
        'close_button': "Fermer",
        'predict_button_text': "Lancer la Prédiction",
        'reset_button_text': "Réinitialiser Formulaire",
        'save_button_text': "Sauvegarder Résultat",
        'info_modal_title': "DiaRisk - Informations",
        'info_modal_accordion_inputs_title': "Paramètres d'Entrée",
        'info_modal_accordion_inputs_p1': "Entrez les métriques standard du patient. Les infobulles fournissent un contexte basé sur les directives cliniques courantes ou les seuils utilisés dans les données d'entraînement du modèle.",
        # Improved French tooltip texts for clarity and consistency with English
        'info_modal_inputs_glucose': "Glucose (mg/dL) : Glycémie 2h post-charge. Tolérance normale : <{normal_max} mg/dL. Tolérance altérée (Prédiabète) : {prediabetes_min}-{prediabetes_max} mg/dL. Diabète : ≥{diabetes_threshold} mg/dL. Exemple : 110",
        'info_modal_inputs_bp': "TA Diastolique (mmHg) : Pression artérielle diastolique. Hypertension : >{hypertension_threshold} mmHg. Hypotension : <{hypotension_threshold} mmHg. Exemple : 75",
        'info_modal_inputs_skinthickness': "Épaisseur Pli Cutané (mm) : Pli tricipital. Normal (réf. femmes) : ≤{normal_threshold} mm. Élevée : >{elevated_threshold} mm. Exemple : 25",
        'info_modal_inputs_insulin': "Insuline (mUI/L) : Insuline sérique 2h post-glucose. Plage normale : {min}-{max} mUI/L. Exemple : 80",
        'info_modal_inputs_bmi': "IMC (kg/m²) : Indice de Masse Corporrelle. Maigreur : <{underweight_threshold}. Normal : {normal_min}-{normal_max}. Surpoids : {overweight_min}-{overweight_max}. Obésité : ≥{obesity_threshold}. Exemple : 28.5",
        'info_modal_inputs_age': "Âge (ans) : Âge du patient. L'âge avancé (ex: >{advanced_age_threshold} ans) est un facteur de risque non modifiable significatif. Exemple : 55. Calculé automatiquement si la date de naissance est fournie.",
        # New Translations for Pregnancies and DPF Info Modal
        'info_modal_inputs_pregnancies': "Grossesses : Nombre de grossesses. Inclut les naissances vivantes, les mortinaissances, les grossesses extra-utérines, etc. Un nombre plus élevé est associé à un risque accru. Exemple : 1",
        'info_modal_inputs_dpf': "Fonction Pedigree Diabète : Une fonction de score génétique qui estime le risque de diabète en fonction des antécédents familiaux. Un score plus élevé indique un risque plus élevé. Exemple : 0.5",
        'input_label_glucose': "Glucose (mg/dL)",
        'input_tooltip_glucose': "Glycémie 2h post-charge. Normal: <140 mg/dL (7.8 mmol/L). Prédiabète: 140-199 mg/dL (7.8-11.0 mmol/L). Diabète: ≥200 mg/dL (11.1 mmol/L). Exemple: 110", # Keep detailed tooltip
        'input_label_bp': "TA Diastolique (mmHg)",
        'input_tooltip_bp': "Pression artérielle diastolique. Hypertension: >90 mmHg. Hypotension: <60 mmHg. Exemple: 75",
        'input_label_skinthickness': "Épaisseur Pli Cutané (mm)", # Kept existing label key
        'input_tooltip_skinthickness_fr': "Pli tricipital. Normal (femmes): ≤23 mm. Élevée: >23 mm. Exemple: 25", # Kept existing tooltip key, text slightly refined
        'input_label_insulin': "Insuline (mUI/L)",
        'input_tooltip_insulin_fr': "Insuline sérique 2h post-glucose. Plage normale : 16-166 mUI/L. Exemple : 80", # Kept existing tooltip key, text slightly refined
        'input_label_bmi': "IMC (kg/m²)",
        'input_tooltip_bmi_fr': "Indice de Masse Corporrelle. Maigreur <18.5. Normal 18.5-24.9. Surpoids 25-29.9. Obésité ≥30. Exemple: 28.5", # Kept existing tooltip key, text slightly refined
        'input_label_age': "Âge (ans)",
        'input_tooltip_age': "L'âge avancé (ex: >65 ans) est un facteur de risque. Exemple : 55. Calculé si date de naissance fournie.", # No specific fr key needed
        # New Translations for Pregnancies and DPF Input Labels/Tooltips
        'input_label_pregnancies': "Grossesses",
        'input_tooltip_pregnancies': "Nombre de grossesses. Exemple : 1",
        'input_label_dpf': "Fonction Pedigree Diabète",
        'input_tooltip_dpf': "Score de risque génétique basé sur les antécédents familiaux. Score plus élevé = risque plus élevé. Exemple : 0.5",
        'card_header_output': "Résultat de la Prédiction du Modèle", # Slightly improved wording
        'card_header_clinical': "Considérations Cliniques Basées sur les Entrées",
        'card_header_graph': "Visualisation des Entrées",
        'clinical_considerations_placeholder': "L'analyse des valeurs d'entrée par rapport aux seuils cliniques apparaîtra ici après la prédiction.",
        'clinical_considerations_header': "Valeurs clés par rapport aux seuils cliniques :",
        'clinical_considerations_none': "Aucune considération clinique spécifique générée basée sur les entrées et seuils définis.",
        'pred_result_low_risk': "Prédiction du Modèle : Faible Risque de Diabète", # Slightly improved wording
        'pred_result_high_risk': "Prédiction du Modèle : Risque Élevé de Diabète", # Slightly improved wording
        'pred_prob_low_risk': "Probabilité (Faible Risque) : {prob:.3f}",
        'pred_prob_high_risk': "Probabilité (Risque Élevé) : {prob:.3f}",
        'alert_model_unavailable': "Modèle de prédiction non disponible.",
        'alert_missing_value': "Valeur manquante pour {name}. Veuillez entrer toutes les valeurs.",
        'alert_value_must_be_positive': "{name} doit être supérieur à zéro.",
        'alert_invalid_age': "Veuillez entrer un âge valide (0-120).",
        'alert_low_bmi': "L'IMC doit être de 10 ou plus.",
        'alert_negative_value': "{name} ne peut pas être négatif.",
        'alert_invalid_numeric': "Valeur numérique invalide pour {name}. Vérifiez les entrées.",
        'alert_scaler_error': "Erreur de prédiction : Problème de composant Scaler.", # Slightly improved wording
        'alert_prediction_error': "Une erreur inattendue s'est produite lors de la prédiction. Vérifiez les logs.",
        # Refined consideration texts
        'consideration_glucose_normal': "**{name} ({val:.1f} {unit}, soit env. {val_mmol:.1f} mmol/L)** : Tolérance normale au glucose.",
        'consideration_glucose_prediabetes': "**{name} ({val:.1f} {unit}, soit env. {val_mmol:.1f} mmol/L)** : Tolérance au glucose altérée (prédiabète). Risque accru de diabète.",
        'consideration_glucose_diabetes': "**{name} ({val:.1f} {unit}, soit env. {val_mmol:.1f} mmol/L)** : Niveau de glucose suggestif de diabète.", # Slightly improved wording
        'consideration_bp_hypertensive': "**{name} ({val:.0f} {unit})** : Indique une pression artérielle diastolique élevée, facteur de risque de diabète de type 2.", # Slightly more specific
        'consideration_bp_hypotensive': "**{name} ({val:.0f} {unit})** : Pression artérielle diastolique basse. Peut être associée à un métabolisme stable ou d'autres facteurs.", # Slightly improved wording
        'consideration_bp_normal': "**{name} ({val:.0f} {unit})** : Pression artérielle diastolique dans la plage normale.",
        'consideration_skinthickness_normal_female': "**{name} ({val:.0f} {unit})** : Épaisseur normale (≤{threshold} mm pour les femmes), suggérant une adiposité saine.", # Slightly improved wording
        'consideration_skinthickness_elevated_female': "**{name} ({val:.0f} {unit})** : Épaisseur élevée (>{threshold} mm), indicateur d'adiposité excessive.", # Slightly improved wording
        'consideration_insulin_normal': "**{name} ({val:.0f} {unit})** : Insuline dans la plage normale (2h post-glucose).", # Slightly improved wording
        'consideration_insulin_high': "**{name} ({val:.0f} {unit})** : Valeur d'insuline élevée (2h post-glucose), peut indiquer une résistance à l'insuline.", # Slightly improved wording
        'consideration_insulin_low': "**{name} ({val:.0f} {unit})** : Valeur d'insuline basse (2h post-glucose), nécessite une évaluation clinique.", # Slightly improved wording
        'consideration_bmi_underweight': "**{name} ({val:.1f} {unit})** : Maigreur.",
        'consideration_bmi_normal': "**{name} ({val:.1f} {unit})** : Poids normal.",
        'consideration_bmi_overweight': "**{name} ({val:.1f} {unit})** : Surpoids (risque modéré de diabète).",
        'consideration_bmi_obesity': "**{name} ({val:.1f} {unit})** : Obésité (risque élevé de diabète).",
        'consideration_age_young': "**{name} ({val:.0f} {unit_age})**: Jeune âge, généralement associé à un risque de base plus faible pour le diabète de type 2.", # Slightly improved wording
        'consideration_age_adult': "**{name} ({val:.0f} {unit_age})**: Âge adulte.",
        'consideration_age_advanced': "**{name} ({val:.0f} {unit_age})**: L'âge avancé est un facteur de risque connu pour le diabète.",
        'warning_positive': "{name} doit être > 0.",
        'warning_bmi_min': "{name} doit être ≥ 10.",
        'warning_age_range': "L'âge doit être entre 0 et 120.",
        'warning_unusual_high': "{name} ({val:.1f}) semble inhabituellement élevé. Veuillez vérifier.", # Added "Veuillez"
        'warning_invalid_format': "Format numérique invalide pour {name}.",
        'warning_future_dob': "La date de naissance ne peut pas être dans le futur.",
        'warning_invalid_date_format': "Format de date invalide. Utilisez AAAA-MM-JJ.",
        'graph_title': "Valeurs Patient vs. Plages de Référence Cliniques",
        'graph_yaxis_label': "Valeur",
        'graph_legend_ref': "Plage de Référence",
        'graph_legend_patient': "Valeur Patient",
        'graph_hover_template': "<b>%{x}</b><br>Valeur Patient : %{y:.1f} %{customdata[2]}<br>Plage Référence : %{customdata[0]:.1f}-{customdata[1]:.1f} %{customdata[2]}<extra></extra>",
        'graph_no_data_title': "Entrez les données patient pour visualiser",
        'graph_no_data_annotation': "Pas de données valides à afficher.",
        'history_page_title': "Historique des Prédictions", # Slightly improved wording
        'history_page_description': "Ce tableau affiche les résultats de prédiction enregistrés précédemment.",
        'history_table_empty': "Aucun historique de prédiction trouvé.",
        'history_header_timestamp': "Horodatage",
        'history_header_glucose': "Glucose",
        'history_header_bp': "TA Diastolique",
        'history_header_skinthickness': "Épaisseur Pli",
        'history_header_insulin': "Insuline",
        'history_header_bmi': "IMC",
        'history_header_age': "Âge",
        # New History Headers for Pregnancies and DPF
        'history_header_pregnancies': "Grossesses",
        'history_header_dpf': "FP Diabète", # Abbreviated for table header (French)
        'history_header_outcome': "Résultat Prédit",
        'history_header_probability': "Probabilité",
        # --- Updated About Model Text (Translated) ---
        'about_model_title': "À propos du Modèle de Prédiction (CatBoost)",
        'about_model_p1': "Cet outil utilise un modèle CatBoost entraîné sur la base de données PIMA Indians Diabetes et des données de 2000 patients à l'Hôpital de Francfort, Allemagne. Il prédit la probabilité de développer un diabète en fonction des caractéristiques d'entrée et fournit un pourcentage de risque.", # Fully translated
        'about_model_p2': "CatBoost est un algorithme de gradient boosting qui utilise des arbres de décision non-biaisés (oblivious decision trees). Il est reconnu pour ses hautes performances, sa robustesse aux hyperparamètres et sa gestion native des caractéristiques catégorielles. Il construit les arbres séquentiellement, où chaque nouvel arbre corrige les erreurs des précédents, en optimisant une fonction de perte différentiable.",
        'about_model_features_title': "Caractéristiques d'Entrée Utilisées par le Modèle :", # Translated
        # Updated to reflect that Pregnancies and DPF are now user inputs
        'about_model_features_list': "Le modèle utilise les caractéristiques d'entrée suivantes collectées directement auprès de l'utilisateur : Grossesses, Glucose, Pression Artérielle Diastolique, Épaisseur du Pli Cutané (Triceps), Insuline (Sérique 2h), IMC, Âge et Fonction Pedigree Diabète. Il intègre également plusieurs caractéristiques calculées en interne (par exemple, ratios, interactions).", # Translated
        # Updated note
        'about_model_data_note': "Note : Le modèle a été principalement entraîné sur des sujets féminins d'origine Pima Indian et une cohorte d'un hôpital allemand. L'applicabilité à d'autres populations peut varier.", # Translated
        'about_model_performance_note_title': "Performance du Modèle :", # Translated
        'about_model_performance_note_text': "Performance approximative validée croisée sur les jeux de données d'entraînement utilisant le modèle CatBoost : Précision (Accuracy) ~100%, Aire sous la courbe ROC (AUC ROC) ~0.99. L'AUC ROC mesure la capacité du modèle à distinguer entre les patients à haut et bas risque selon différents seuils de décision, des valeurs plus proches de 1 indiquant une meilleure performance. Ces métriques sont indicatives et la performance réelle peut différer dans des scénarios cliniques spécifiques.", # Translated
        'about_model_disclaimer': "Ce modèle fournit une probabilité de risque basée sur les caractéristiques d'entrée et est destiné à compléter, et non à remplacer, le jugement clinique posé par des professionnels de santé qualifiés.", # Translated
         # --- Keys for translating history table content ---
        'history_outcome_low': "Faible Risque", # Internal key stored in CSV -> Display text
        'history_outcome_high': "Risque Élevé", # Internal key stored in CSV -> Display text
         # PDF Headers for new inputs
        'pdf_pregnancies_header': "Grossesses",
        'pdf_dpf_header': "FP Diabète", # Abbreviated for PDF table header (French)
    },
  'ar': {
        'first_name_label': "الاسم الأول",
        'last_name_label': "اسم العائلة",
        'dob_label': "تاريخ الميلاد",
        'first_name_placeholder': "أدخل الاسم الأول للمريض",
        'last_name_placeholder': "أدخل اسم عائلة المريض",
        'dob_placeholder': "اختر تاريخ الميلاد",
        'pdf_first_name_header': "الاسم الأول",
        'pdf_last_name_header': "اسم العائلة",
        'pdf_dob_header': "تاريخ الميلاد",
        'history_header_first_name': "الاسم الأول",
        'history_header_last_name': "اسم العائلة",
        'history_header_dob': "تاريخ الميلاد",
        'save_pdf_button_text': "حفظ التقرير (PDF)",
        'pdf_report_title': "DiaRisk - تقرير التحليل الطبي",
        'pdf_generated_on': "تم إنشاء التقرير في: {timestamp}",
        'pdf_patient_data_header': "بيانات إدخال المريض",
        'pdf_parameter_header': "المعلمة",
        'pdf_value_header': "القيمة",
        'pdf_unit_header': "الوحدة",
        'pdf_prediction_header': "نتيجة تنبؤ النموذج",
        'pdf_considerations_header': "الاعتبارات السريرية بناءً على المدخلات",
        'pdf_no_considerations': "لم يتم إنشاء اعتبارات سريرية محددة بناءً على المدخلات والعتبات المحددة.",
        'alert_pdf_generation_error': "خطأ في إنشاء تقرير PDF. يرجى التحقق من السجلات.",
        'alert_pdf_disabled': "تتطلب ميزة إنشاء PDF مكتبة 'reportlab'.",
        'alert_pdf_arabic_tools_missing': "تتطلب ميزة إنشاء PDF للغة العربية أدوات إضافية (arabic_reshaper، python-bidi) وخط عربي. يرجى تثبيتها.",
        'alert_pdf_no_clinical_data': "يرجى إدخال البيانات السريرية للمريض وتشغيل التنبؤ أولاً لإنشاء تقرير ذي معنى.",
        'app_title': "DiaRisk - تقييم مخاطر السكري (السريري)",
        'sidebar_header': " DiaRisk", # Keep as is, it's the app name/branding
        'sidebar_home': "الرئيسية",
        'sidebar_predictor': "متنبئ المخاطر",
        'sidebar_history': "سجل المرضى",
        'sidebar_about_model': "حول النموذج",
        'theme_switch_label': "الوضع الداكن",
        'close_button': "إغلاق",
        'predict_button_text': "تشغيل التنبؤ",
        'reset_button_text': "إعادة تعيين النموذج",
        'save_button_text': "حفظ النتيجة",
        'info_modal_title': "DiaRisk - معلومات",
        'info_modal_accordion_inputs_title': "معلمات الإدخال",
        'info_modal_accordion_inputs_p1': "أدخل مقاييس المريض القياسية. توفر تلميحات الأدوات سياقًا يعتمد على الإرشادات السريرية الشائعة أو العتبات المستخدمة في بيانات تدريب النموذج.",
        'info_modal_inputs_glucose': "الجلوكوز (ملغ/ديسيلتر): جلوكوز البلازما بعد ساعتين من التحميل. التحمل الطبيعي: <{normal_max} ملغ/ديسيلتر. ضعف التحمل (ما قبل السكري): {prediabetes_min}-{prediabetes_max} ملغ/ديسيلتر. السكري: ≥{diabetes_threshold} ملغ/ديسيلتر. مثال: 110",
        'info_modal_inputs_bp': "ضغط الدم الانبساطي (مم زئبق): ضغط الدم الانبساطي. ارتفاع ضغط الدم: >{hypertension_threshold} مم زئبق. انخفاض ضغط الدم: <{hypotension_threshold} مم زئبق. مثال: 75",
        'info_modal_inputs_skinthickness': "سمك الجلد (مم): ثنية الجلد ثلاثية الرؤوس. طبيعي (للنساء): ≤{normal_threshold} مم. مرتفع: >{elevated_threshold} مم. مثال: 25",
        'info_modal_inputs_insulin': "الأنسولين (mUI/L): أنسولين المصل بعد ساعتين. طبيعي: {min}-{max} وحدة دولية صغيرة/لتر. مثال: 80",
        'info_modal_inputs_bmi': "مؤشر كتلة الجسم (كجم/م²): مؤشر كتلة الجسم. نقص الوزن <{underweight_threshold}. طبيعي {normal_min}-{normal_max}. زيادة الوزن {overweight_min}-{overweight_max}. سمنة ≥{obesity_threshold}. مثال: 28.5",
        'info_modal_inputs_age': "العمر (سنوات): عمر المريض. العمر المتقدم (مثال: >{advanced_age_threshold} سنة) عامل خطر غير قابل للتعديل. مثال: 55. يُحسب تلقائيًا إذا تم توفير تاريخ الميلاد.",
         # New Translations for Pregnancies and DPF Info Modal
        'info_modal_inputs_pregnancies': "الحمل : عدد مرات الحمل. يشمل الولادات الحية، والإملاص، والحمل خارج الرحم، وما إلى ذلك. الأعداد الأعلى ترتبط بزيادة الخطر. مثال: 1",
        'info_modal_inputs_dpf': "دالة النسب السكري : دالة نقاط وراثية تقدر خطر الإصابة بالسكري بناءً على التاريخ العائلي. النتيجة الأعلى تشير إلى خطر أعلى. مثال: 0.5",
        'select_language': 'اختر اللغة', # Minor adjustment for consistency
        'current_language': 'اللغة الحالية: {lang}',
        'legal_disclaimer': "هذا التطبيق لا يحل محل الاستشارة الطبية. للاستخدام من قبل أخصائيي الرعاية الصحية المؤهلين.", # Corrected for flow
        'landing_title': "DiaRisk - تقييم مخاطر السكري",
        'landing_subtitle': "دعم التعلم الآلي لتقسيم المخاطر السريرية.",
        'landing_button': "فتح المتنبئ",
        'landing_error_title': "خطأ في التطبيق",
        'landing_error_message': "خطأ في تحميل مكونات التنبؤ: {error}",
        'landing_error_suggestion': "يرجى التأكد من وجود ملفات النموذج والمقياس وإمكانية الوصول إليها، أو التحقق من سجلات الخادم لمزيد من التفاصيل.",
        'dashboard_title': "DiaRisk - المتنبئ",
        'card_header_input': "إدخال بيانات المريض",
        'input_label_glucose': "الجلوكوز (ملغ/ديسيلتر)",
        'input_tooltip_glucose': "بعد ساعتين من التحميل. طبيعي: <140. ما قبل السكري: 140-199. السكري: ≥200. مثال: 110",
        'input_label_bp': "ضغط الدم الانبساطي (مم زئبق)",
        'input_tooltip_bp': "ارتفاع ضغط الدم: >90. انخفاض ضغط الدم: <60. مثال: 75",
        'input_label_skinthickness': "سمك الجلد (مم)",
        'input_tooltip_skinthickness_ar': "ثنية الجلد ثلاثية الرؤوس. طبيعي (للنساء): ≤23 مم. مرتفع: >23 مم. مثال: 25",
        'input_label_insulin': "الأنسولين (mUI/L)",
        'input_tooltip_insulin_ar': "أنسولين المصل بعد ساعتين. طبيعي: 16-166 وحدة دولية صغيرة/لتر. مثال: 80",
        'input_label_bmi': "مؤشر كتلة الجسم (كجم/م²)",
        'input_tooltip_bmi_ar': "نقص الوزن <18.5. طبيعي 18.5-24.9. زيادة الوزن 25-29.9. سمنة ≥30. مثال: 28.5",
        'input_label_age': "العمر (سنوات)",
        'input_tooltip_age': "العمر المتقدم (مثال: >65) عامل خطر. مثال: 55. يُحسب تلقائيًا إذا تم توفير تاريخ الميلاد.",
        # New Translations for Pregnancies and DPF Input Labels/Tooltips
        'input_label_pregnancies': "الحمل",
        'input_tooltip_pregnancies': "عدد مرات الحمل. مثال: 1",
        'input_label_dpf': "دالة النسب السكري",
        'input_tooltip_dpf': "نقاط الخطر الوراثي بناءً على التاريخ العائلي. نقاط أعلى = خطر أعلى. مثال: 0.5",
        'card_header_output': "مخرجات تنبؤ النموذج",
        'card_header_clinical': "الاعتبارات السريرية بناءً على المدخلات",
        'card_header_graph': "تصور المدخلات",
        'clinical_considerations_placeholder': "سيظهر تحليل قيم الإدخال بالنسبة للعتبات السريرية هنا بعد التنبؤ.",
        'clinical_considerations_header': "قيم الإدخال الرئيسية بالنسبة للعتبات السريرية:",
        'clinical_considerations_none': "لم يتم إنشاء اعتبارات سريرية محددة بناءً على المدخلات والعتبات المحددة.",
        'pred_result_low_risk': "تنبؤ النموذج: خطر منخفض للإصابة بالسكري",
        'pred_result_high_risk': "تنبؤ النموذج: خطر مرتفع للإصابة بالسكري",
        'pred_prob_low_risk': "الاحتمالية (خطر منخفض): {prob:.3f}",
        'pred_prob_high_risk': "الاحتمالية (خطر مرتفع): {prob:.3f}",
        'alert_model_unavailable': "نموذج التنبؤ غير متوفر.",
        'alert_missing_value': "قيمة مفقودة لـ {name}. الرجاء إدخال جميع القيم.",
        'alert_value_must_be_positive': "يجب أن يكون {name} أكبر من صفر.",
        'alert_invalid_age': "الرجاء إدخال عمر صالح (0-120).",
        'alert_low_bmi': "يجب أن يكون مؤشر كتلة الجسم 10 أو أكثر.",
        'alert_negative_value': "لا يمكن أن يكون {name} سالبًا.",
        'alert_invalid_numeric': "قيمة رقمية غير صالحة لـ {name}. يرجى التحقق من المدخلات.",
        'alert_scaler_error': "خطأ في التنبؤ: مشكلة في مكون المقياس.",
        'alert_prediction_error': "حدث خطأ غير متوقع أثناء التنبؤ. تحقق من السجلات.",
        'consideration_glucose_normal': "**{name} ({val:.1f} {unit})**: تحمل طبيعي للجلوكوز.",
        'consideration_glucose_prediabetes': "**{name} ({val:.1f} {unit})**: ضعف تحمل الجلوكوز (ما قبل السكري). زيادة خطر الإصابة بالسكري.",
        'consideration_glucose_diabetes': "**{name} ({val:.1f} {unit})**: مستوى جلوكوز يشير لاحتمالية السكري.",
        'consideration_bp_hypertensive': "**{name} ({val:.0f} {unit})**: يشير إلى ارتفاع ضغط الدم الانبساطي، وهو عامل خطر لمرض السسكري من النوع الثاني.",
        'consideration_bp_hypotensive': "**{name} ({val:.0f} {unit})**: انخفاض ضغط الدم الانبساطي. قد يكون مرتبطًا باستقلاب أكثر استقرارًا أو عوامل أخرى.",
        'consideration_bp_normal': "**{name} ({val:.0f} {unit})**: ضغط الدم الانبساطي في النطاق الطبيعي.",
        'consideration_skinthickness_normal_female': "**{name} ({val:.0f} {unit})**: سمك طبيعي (≤{threshold} مم للإناث)، مما يشير إلى دهون الجسم الصحية.",
        'consideration_skinthickness_elevated_female': "**{name} ({val:.0f} {unit})**: سمك مرتفع (>{threshold} مم)، علامة على السمنة المفرطة.",
        'consideration_insulin_normal': "**{name} ({val:.0f} {unit})**: الأنسولين ضمن النطاق الطبيعي (بعد ساعتين من الجلوكوز).",
        'consideration_insulin_high': "**{name} ({val:.0f} {unit})**: قيمة أنسولين مرتفعة (بعد ساعتين)، قد تشير إلى مقاومة الأنسولين.",
        'consideration_insulin_low': "**{name} ({val:.0f} {unit})**: قيمة أنسولين منخفضة (بعد ساعتين)، تتطلب تقييمًا سريريًا.",
        'consideration_bmi_underweight': "**{name} ({val:.1f} {unit})**: نقص الوزن.",
        'consideration_bmi_normal': "**{name} ({val:.1f} {unit})**: وزن طبيعي.",
        'consideration_bmi_overweight': "**{name} ({val:.1f} {unit})**: زيادة الوزن (خطر معتدل للإصابة بالسكري).",
        'consideration_bmi_obesity': "**{name} ({val:.1f} {unit})**: السمنة (خطر كبير للإصابة بالسكري).",
        'consideration_age_young': "**{name} ({val:.0f} {unit_age})**: عمر صغير، يرتبط عادة بخطر أساسي أقل لمرض السكري من النوع الثاني.",
        'consideration_age_adult': "**{name} ({val:.0f} {unit_age})**: عمر بالغ.",
        'consideration_age_advanced': "**{name} ({val:.0f} {unit_age})**: العمر المتقدم عامل خطر معروف لمرض السكري.",
        'warning_positive': "يجب أن يكون {name} أكبر من صفر.",
        'warning_bmi_min': "يجب أن يكون مؤشر كتلة الجسم أكبر من أو يساوي 10.",
        'warning_age_range': "يجب أن يكون العمر بين 0 و 120 سنة.",
        'warning_unusual_high': "{name} ({val:.1f}) يبدو مرتفعًا بشكل غير عادي. يرجى التحقق.",
        'warning_invalid_format': "تنسيق رقمي غير صالح لـ {name}.",
        'warning_future_dob': "لا يمكن أن يكون تاريخ الميلاد في المستقبل.",
        'warning_invalid_date_format': "تنسيق تاريخ غير صالح. استخدم AAAA-MM-JJ.",
        'graph_title': "قيم المريض مقابل النطاقات المرجعية السريرية",
        'graph_yaxis_label': "القيمة",
        'graph_legend_ref': "النطاق المرجعي",
        'graph_legend_patient': "قيمة المريض",
        'graph_hover_template': "<b>%{x}</b><br>قيمة المريض: %{y:.1f} %{customdata[2]}<br>النطاق المرجعي: %{customdata[0]:.1f}-{customdata[1]:.1f} %{customdata[2]}<extra></extra>",
        'graph_no_data_title': "أدخل بيانات المريض للتصور",
        'graph_no_data_annotation': "لا توجد بيانات صالحة للعرض.",
        'history_page_title': "سجل تاريخ التنبؤات",
        'history_page_description': "يعرض هذا الجدول نتائج التنبؤ المسجلة مسبقًا.",
        'history_table_empty': "لم يتم العثور على سجل تنبؤات.",
        'history_header_timestamp': "الطابع الزمني",
        'history_header_glucose': "الجلوكوز",
        'history_header_bp': "ضغط الدم الانبساطي",
        'history_header_skinthickness': "سمك الجلد",
        'history_header_insulin': "الأنسولين",
        'history_header_bmi': "مؤشر كتلة الجسم",
        'history_header_age': "العمر",
        # New History Headers for Pregnancies and DPF
        'history_header_pregnancies': "الحمل",
        'history_header_dpf': "دالة النسب", # Abbreviated for table header (Arabic)
        'history_header_outcome': "النتيجة المتوقعة",
        'history_header_probability': "الاحتمالية",
        'about_model_title': "حول نموذج التنبؤ (CatBoost)",
        'about_model_p1': "تستخدم هذه الأداة نموذج CatBoost المدرب على قاعدة بيانات PIMA Indians Diabetes وبيانات من 2000 مريض في مستشفى فرانكفورت بألمانيا. يتنبأ النموذج باحتمالية الإصابة بمرض السكري بناءً على ميزات الإدخال ويوفر احتمال الخطر.",
        'about_model_p2': "CatBoost هي خوارزمية تعزيز التدرج تستخدم أشجار القرار المتغافلة. وهي معروفة بأدائها العالي وقوتها ضد المعلمات الفائقة ومعالجتها الأصلية للميزات الفئوية. تقوم ببناء الأشجار بشكل تسلسلي، حيث تصحح كل شجرة جديدة أخطاء الأشجار السابقة، وتحسين دالة خسارة قابلة للتفاضل.",
        'about_model_features_title': "ميزات الإدخال المستخدمة بواسطة النموذج :",
        # Updated to reflect that Pregnancies and DPF are now user inputs
        'about_model_features_list': "يستخدم النموذج ميزات الإدخال التالية التي تم جمعها مباشرة من المستخدم: الحمل، الجلوكوز، ضغط الدم الانبساطي، سمك الجلد (ثلاثية الرؤوس)، الأنسولين (مصل الدم بعد ساعتين)، مؤشر كتلة الجسم، العمر، ودالة النسب السكري. كما يتضمن عدة ميزات محسوبة داخليًا مشتقة من هذه المدخلات (مثل النسب والتفاعلات).",
         # Updated note
        'about_model_data_note': "ملاحظة : تم تدريب النموذج بشكل أساسي على الإناث من أصل Pima الهندي ومجموعة من المرضى من مستشفى ألماني. قد تختلف قابلية التطبيق على المجموعات السكانية الأخرى.",
        'about_model_performance_note_title': "أداء النموذج :",
        'about_model_performance_note_text': "الأداء التقريبي المعاير تبادليًا على مجموعات بيانات التدريب باستخدام نموذج CatBoost: الدقة (Accuracy) ~100%، المساحة تحت منحنى ROC (AUC ROC) ~0.99. تقيس AUC ROC قدرة النموذج على التمييز بين المرضى ذوي الخطر المرتفع والمنخفض عبر عتبات قرار مختلفة، حيث تشير القيم الأقرب إلى 1 إلى أداء أفضل. هذه المقاييس إرشادية وقد يختلف الأداء الفعلي في السيناريوهات السريرية المحددة.",
        'about_model_disclaimer': "يقدم هذا النموذج احتمالية خطر بناءً على ميزات الإدخال وهو مخصص لتكملة، وليس استبدال، الحكم السريري من قبل المتخصصين المؤهلين في الرعاية الصحية.",
        'history_outcome_low': "خطر منخفض",
        'history_outcome_high': "خطر مرتفع",
         # PDF Headers for new inputs
        'pdf_pregnancies_header': "الحمل",
        'pdf_dpf_header': "دالة النسب", # Abbreviated for PDF table header (Arabic)
    }
}

DEFAULT_LANG = 'en'

def get_translation(lang, key, **kwargs):
    effective_lang = lang if lang in TRANSLATIONS else DEFAULT_LANG
    lang_dict = TRANSLATIONS.get(effective_lang, TRANSLATIONS[DEFAULT_LANG])
    # Added check for key existence in default as fallback
    text_template = lang_dict.get(key, TRANSLATIONS[DEFAULT_LANG].get(key, f"[{key}]"))
    if key == 'graph_hover_template':
        return text_template
    try:
        return text_template.format(**kwargs)
    except KeyError as e: # pragma: no cover
        logger.warning(f"Translation key '{key}' for lang '{effective_lang}' is missing placeholder: {e}. Template: '{text_template}'")
        missing_key = str(e).strip("'")
        # Attempt to format skipping the missing key, adding a placeholder string
        safe_template = text_template.replace("{" + missing_key + "}", f"[MISSING_VAR:{missing_key}]")
        try:
             return safe_template.format(**{k: v for k, v in kwargs.items() if k != missing_key})
        except Exception:
             # If even safe formatting fails, return the original template with error note
             return f"[{key} - Format Error: {e}]"
    except Exception as e: # pragma: no cover
        logger.error(f"Error formatting translation for key '{key}', lang '{effective_lang}': {e}", exc_info=True)
        return f"[{key} - Format Error]"


def create_default_graph(lang=DEFAULT_LANG, theme='light'):
    plotly_template = "plotly_dark" if theme == 'dark' else "plotly_white"
    is_dark = theme == 'dark'
    fig = go.Figure()
    fig.update_layout(
        template=plotly_template,
        xaxis={'visible': False}, yaxis={'visible': False},
        annotations=[{
            'text': get_translation(lang, 'graph_no_data_annotation'),
            'xref': 'paper', 'yref': 'paper', 'x': 0.5, 'y': 0.5, 'showarrow': False,
            'font': {'size': 14, 'color': '#adb5bd' if is_dark else '#6c757d'}
        }],
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


app = dash.Dash(__name__,
                external_stylesheets=[dbc.icons.FONT_AWESOME, light_theme_url], # Start with light theme
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
                assets_folder='assets')
server = app.server

# --- app.index_string with corrected JS comments and proper closing quotes ---
app.index_string = '''
<!DOCTYPE html>
<html lang="en" dir="ltr">
    <head>
        {%metas%}
        <title>DiaRisk - Diabetes Risk Assessment</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
        <style>
            html, body { height: 100%; margin: 0; padding: 0; overflow-x: hidden; }
            #app-container { position: relative; min-height: 100vh; transition: background-color 0.3s ease, color 0.3s ease; display: flex; flex-direction: column;}
            #main-content-wrapper { flex-grow: 1; display: flex; }
            @keyframes gradient{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
            @keyframes slideIn{0%{transform:translateY(50px);opacity:0}100%{transform:translateY(0);opacity:1}}
            @keyframes pulse{0%{transform:scale(1)}50%{transform:scale(1.05)}100%{transform:scale(1)}}
            @keyframes resultFadeIn{0%{opacity:0;transform:scale(.8)}100%{opacity:1;transform:scale(1)}}
            .input-container{position:relative;margin-bottom:1.5rem}
            .input-container .form-control{width:100%; padding:10px 12px; border: 2px solid #dee2e6; border-radius:8px; font-size:1rem; transition:all .3s ease; height:auto; background-clip: padding-box; }
            .input-container .form-control:focus{outline:none; border-color:#86b7fe; box-shadow:0 0 0 .25rem rgba(13,110,253,.25)}
            .input-container input[type=number]{-moz-appearance:textfield} .input-container input[type=number]::-webkit-outer-spin-button,.input-container input[type=number]::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
            .input-container input[type="date"].form-control { padding-top: 10px; padding-bottom: 10px; }
            .input-container input[type="date"].form-control:focus,
            .input-container input[type="date"].form-control:not(:placeholder-shown),
            .input-container input[type="date"].form-control:-webkit-autofill { padding-top: 16px; padding-bottom: 4px; }
            .input-container label{position:absolute; top:12px; left:12px; font-size:1rem; color:#6c757d; transition:all .2s ease-out; pointer-events:none; padding:0 5px; z-index:1; background: white;}
            body.rtl .input-container label { right: 12px; left: auto; } /* RTL */
            .input-container .form-control:focus + label, .input-container .form-control:not(:placeholder-shown) + label, .input-container .form-control:-webkit-autofill + label {top:-10px; left:10px; font-size:.8rem; color:#0d6efd; font-weight:bold; z-index:1}
            body.rtl .input-container .form-control:focus + label, body.rtl .input-container .form-control:not(:placeholder-shown) + label, body.rtl .input-container .form-control:-webkit-autofill + label { right: 10px; left: auto; } /* RTL */
            .input-container .form-control:focus:not([type="date"]), .input-container .form-control:not(:placeholder-shown):not([type="date"]), .input_container .form-control:-webkit-autofill:not([type="date"]) {padding-top:16px;padding-bottom:4px}
            .predict-button, .reset-button, .save-pdf-button {border-radius:25px;padding:10px 20px; font-size:1rem; transition:all .3s ease; margin-top: 0.5rem;}
            .predict-button{background:linear-gradient(45deg,#0d6efd,#6610f2);border:none; color:white; width: auto; flex-grow: 1;}
            .predict-button:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(0,0,0,.2)} .predict-button:active{transform:translateY(0);box-shadow:0 2px 6px rgba(0,0,0,.15)}
            .predict-button:disabled { background: #adb5bd; cursor: not-allowed; opacity: 0.65; transform: none; box-shadow: none;}
            .reset-button { width: auto;}
            .save-pdf-button { width: auto; background-color: #198754; color: white; border: none;}
             .save-pdf-button:hover { background-color: #157347; transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,.15); }
             .save_pdf_button:disabled { background: #adb5bd; cursor: not-allowed; opacity: 0.65; transform: none; box-shadow: none; }
            .button-wrapper { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap;}
            body.rtl .button-wrapper { justify-content: flex-end; } /* RTL */
            .result-container{animation:resultFadeIn .5s ease-out; padding:20px; border-radius:10px; background:#f0f0f0; box-shadow:inset 0 2px 4px rgba(0,0,0,.06); margin-top:1.5rem; transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;}
            .sidebar { position: fixed; top: 0; left: 0; bottom: 0; width: ''' + SIDEBAR_WIDTH_EXPANDED + '''; padding: 1rem; background-color: #f8f9fa; border-right: 1px solid #dee2e6; transition: width 0.3s ease, background-color 0.3s ease, border-color 0.3s ease, left 0s, right 0s; z-index: 1030; overflow-y: auto; overflow-x: hidden; display: flex; flex-direction: column; }
            body.rtl .sidebar { left: auto; right: 0; border-right: none; border-left: 1px solid #dee2e6; } /* RTL */
            body.rtl.dark-theme .sidebar { border-left-color: #495057; border-right: none;} /* RTL */
            .sidebar .nav-link { color:#495057; padding:.75rem 1rem; border-radius:.3rem; transition:background-color .2s ease,color .2s ease; white-space:nowrap; overflow:hidden; text-overflow: ellipsis; display: flex; align-items: center; }
            body.rtl .sidebar .nav-link { flex-direction: row-reverse; } /* RTL */
            .sidebar .nav-link .nav-text { margin-left: 0.75rem; transition: opacity 0.2s ease; opacity: 1; }
             body.rtl .sidebar .nav-link .nav-text { margin-left: 0; margin-right: 0.75rem; } /* RTL */
            .sidebar-header{ font-size:1.5rem; font-weight:bold; color:#343a40; margin-bottom:1.5rem; padding-bottom:.5rem; border-bottom:1px solid #dee2e6; display:flex; align-items:center; white-space:nowrap; overflow:hidden; transition: color 0.3s ease, border-color 0.3s ease; }
            body.rtl .sidebar-header { flex-direction: row-reverse; } /* RTL */
            .sidebar-header .fa-laptop-medical{margin-right:.75rem; color:#0d6efd; transition: margin-right 0.3s ease;}
            body.rtl .sidebar-header .fa-laptop-medical { margin-right: 0; margin-left: 0.75rem;} /* RTL */
            .sidebar-header .header-text { transition: opacity 0.2s ease; opacity: 1; }
            .sidebar.collapsed { width: ''' + SIDEBAR_WIDTH_COLLAPSED + '''; }
            .sidebar.collapsed .nav-link .nav-text { opacity: 0; width: 0; margin-left: 0; margin-right: 0; pointer-events: none; }
            .sidebar.collapsed .nav-link { justify-content: center; }
            .sidebar.collapsed .sidebar-header .fa-laptop-medical { margin-right: 0; margin-left:0; }
            .sidebar.collapsed .sidebar-header .header-text { opacity: 0; width: 0; pointer-events: none; }
            .sidebar.collapsed .sidebar-header { justify-content: center; }
            .sidebar.collapsed .theme-switch-wrapper label { display: none; }
            .sidebar.collapsed .theme-switch-wrapper { justify-content: center; padding: 0.5rem 0; }
            .sidebar.collapsed .language-selector-wrapper label { display: none; }
            .sidebar-toggle-button { margin-top: auto; margin-bottom: 1rem; width: 100%; text-align: center; background: rgba(0,0,0,0.05); border: none; color: #6c757d; padding: 0.5rem 0; border-radius: 0.3rem; }
            .sidebar-toggle-button:hover { background: rgba(0,0,0,0.1); color: #0d6efd; }
            .language-selector-wrapper { padding: 0 1rem; margin-bottom: 1rem; }
            .language-selector-wrapper label { display: block; margin-bottom: 0.3rem; font-size:0.9rem; font-weight:500; color: #495057; transition: color 0.3s ease, opacity 0.2s ease; opacity: 1;}
            .language-selector-wrapper .Select-control { font-size: 0.9rem; border-radius: 0.3rem; border: 1px solid #ced4da; }
            .language-selector-wrapper .Select-menu-outer { z-index: 1060; }
            body.rtl .language-selector-wrapper { text-align: right; } /* RTL */
            #page-content { transition: margin-left 0.3s ease, margin-right 0.3s ease, background-color 0.3s ease, color 0.3s ease; background-color: inherit; padding: 1rem; width: 100%; flex-grow: 1; }
            @media (min-width: 768px) { #page-content { padding: 2rem 1rem; } }
            .info-button{position:absolute;top:15px;right:15px;font-size:1.5rem;color:#6c757d;z-index:1000;background:none;border:none;padding:.25rem .5rem; transition: color 0.3s ease, right 0s, left 0s;} .info-button:hover{color:#0d6efd}
            body.rtl .info-button { right: auto; left: 15px; } /* RTL */
            .warning-text{font-size:.85rem;color:#dc3545;margin-top:-1rem;margin-bottom:1rem;min-height:1.2em}
            .loading-state{text-align:center;padding:20px}
            .clinical-considerations-list { padding-left: 1rem; list-style: none;}
            body.rtl .clinical-considerations-list { padding-left: 0; padding-right: 1rem;} /* RTL */
            .clinical-considerations-list li{position: relative; margin-bottom:.75rem; padding-left:.75rem; }
            body.rtl .clinical-considerations-list li{ padding-left: 0; padding-right:.75rem; } /* RTL */
            .clinical-considerations-list li::before { content: ''; position: absolute; left: -0.5rem; top: 0.4em; width: 6px; height: 6px; border-radius: 50%; background-color: #6c757d; transition: background-color 0.3s ease; }
            body.rtl .clinical-considerations-list li::before { left: auto; right: -0.5rem; } /* RTL */
            .clinical-considerations-list li.highlight::before{background-color:#dc3545;}
            .clinical-considerations-list li.moderate-alert::before{background-color:#ffc107;}
            .clinical-considerations-list li.normal::before{background-color:#198754;}
            .clinical-considerations-list li strong{color:#343a40; transition: color 0.3s ease;}
            .clinical-considerations-list li p { margin-bottom: 0; }
            .theme-switch-wrapper { display: flex; align-items: center; justify-content: space-between; padding: 0.5rem 1rem; margin-top: 1rem; background-color: rgba(0,0,0,0.03); border-radius: 0.3rem; transition: background-color 0.3s ease;}
            .theme-switch-wrapper label { margin-bottom: 0; color: #495057; transition: color 0.3s ease, opacity 0.2s ease; opacity: 1;}
            .theme-switch-wrapper .form-switch .form-check-input { cursor: pointer; width: 3em; height: 1.5em;}
            body.rtl .theme-switch-wrapper { flex-direction: row-reverse; } /* RTL */
            .app-footer { padding: 0.8rem 1rem; background-color: #e9ecef; border-top: 1px solid #dee2e6; text-align: center; font-size: 0.85rem; color: #6c757d; font-style: italic; margin-top: auto; transition: margin-left 0.3s ease, margin-right 0.3s ease, background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease; flex-shrink: 0; }
            body.dark-theme .app-footer { background-color: #343a40; border-top-color: #495057; color: #adb5bd; }
            body.dark-theme { background-color: #222; color: #dee2e6; }
            body.dark-theme #page-content { background-color: #222; }
            .dark-theme .card { background-color: #343a40; border-color: #495057; color: #dee2e6; }
            .dark-theme .card-header { background-color: #495057; border-bottom-color: #6c757d; color: #f8f9fa;}
            .dark-theme .modal-content { background-color: #343a40; color: #dee2e6; border-color: #495057; }
            .dark-theme .modal-header, .dark-theme .modal-footer { border-color: #495057; }
            .dark_theme .accordion-item { background-color: #343a40; border-color: #495057; }
            .dark-theme .accordion-button { background-color: transparent; color: #f8f9fa; box-shadow: none !important; }
            .dark-theme .accordion-button:not(.collapsed) { background-color: rgba(255,255,255,0.05); color: #f8f9fa;}
            body.rtl .accordion-button::after { margin-left: 0; margin-right: auto; } /* RTL */
            .dark-theme .accordion-button::after { filter: invert(1) grayscale(100%) brightness(200%); }
            .dark-theme .accordion-body { color: #adb5bd; }
            .dark-theme .list-group-item { background-color: #343a40; border-color: #495057; color: #dee2e6; }
            .dark-theme .alert-success { background-color: #146c43; border-color: #198754; color: #fff; }
            .dark-theme .alert-danger { background-color: #b02a37; border-color: #dc3545; color: #fff; }
            .dark-theme .alert-warning { background-color: #b97d10; border-color: #ffc107; color: #000; }
            .dark-theme .text-muted { color: #adb5bd !important; }
            .dark-theme h1, .dark-theme h2, .dark-theme h3, .dark-theme h4, .dark-theme h5, .dark-theme h6 { color: #f8f9fa; }
            .dark-theme .sidebar { background-color: #212529; border-right-color: #495057; border-left-color: #495057; }
            .dark-theme .sidebar .nav-link { color: #adb5bd; }
            .dark-theme .sidebar .nav-link:hover { background-color: #343a40; color: #dee2e6; }
            .dark-theme .sidebar .nav-link.active { background-color: #0d6efd; color: #fff; }
            .dark-theme .sidebar-header { color: #f8f9fa; border-bottom-color: #495057; }
            .dark-theme .sidebar-header .fa-laptop-medical { color: #3b82f6; }
            .dark-theme .info-button { color: #adb5bd; } .dark-theme .info-button:hover{color:#3b82f6;}
            .dark-theme .result-container { background-color: #495057; color: #dee2e6; box-shadow: inset 0 2px 4px rgba(0,0,0,.25); }
            .dark-theme .clinical-considerations-list li::before { background_color: #adb5bd;}
            .dark-theme .clinical-considerations-list li.highlight::before { background-color: #ff6b6b; }
            .dark_theme .clinical_considerations_list li.moderate-alert::before { background-color: #facc15; }
            .dark-theme .clinical-considerations-list li.normal::before { background-color: #4ade80; }
            .dark_theme .clinical-considerations-list li strong { color: #f8f9fa; }
            .dark_theme .theme-switch-wrapper { background_color: rgba(255,255,255,0.1); }
            .dark_theme .theme-switch-wrapper label { color: #adb5bd; }
            .dark-theme .language-selector-wrapper label { color: #adb5bd; }
            .dark-theme .language-selector-wrapper .Select-control { background-color: #343a40; border-color: #6c757d; }
            .dark-theme .language-selector-wrapper .Select-value-label, .dark_theme .language-selector-wrapper .Select-placeholder { color: #dee2e6; }
            .dark_theme .language-selector-wrapper .Select--single > .Select-control .Select-arrow { border-top-color: #adb5bd;}
            .dark-theme .language-selector-wrapper .Select-menu-outer { background-color: #343a40; border-color: #495057; }
            .dark-theme .language-selector-wrapper .Select-option { color: #adb5bd; }
            .dark_theme .language-selector-wrapper .Select-option.is-focused { background-color: rgba(255,255,255,0.1); color: #f8f9fa; }
            .dark-theme .language-selector-wrapper .Select-option.is-selected { background-color: rgba(13, 110, 253, 0.5); color: #fff; }
            .dark-theme .input-container label { color: #adb5bd; background: #343a40; }
            .dark-theme .input-container .form-control { background-color: #343a40; border-color: #6c757d; color: #dee2e6; }
            .dark-theme .input-container input[type="date"].form-control { color-scheme: dark; }
            .dark-theme .input-container .form-control::placeholder { color: #6c757d; opacity: 1; }
            .dark-theme .input-container .form-control:focus { background-color: #343a40; border-color: #86b7fe; color: #dee2e6; box-shadow: 0 0 0 .25rem rgba(59, 130, 246, 0.25); }
            .dark-theme .input-container .form-control:focus + label, .dark-theme .input-container .form-control:not(:placeholder-shown) + label, .dark-theme .input-container .form-control:-webkit-autofill + label { color: #86b7fe; background: #343a40; }
            body.rtl.dark-theme .input-container label { background: #343a40; } /* RTL */
            body.rtl.dark-theme .input-container .form-control:focus + label, body.rtl.dark-container .form-control:not(:placeholder-shown) + label, body.rtl.dark-container .form-control:-webkit-autofill + label { background: #343a40; } /* RTL */
            .dark-theme .sidebar-toggle-button { background: rgba(255,255,255,0.1); color: #adb5bd;}
            .dark_theme .sidebar-toggle-button:hover { background: rgba(255,255,255,0.2); color: #dee2e6;}
            .dark-theme .save-pdf-button { background-color: #20c997; color: #fff; }
            .dark-theme .save_pdf_button:hover { background_color: #1baa80; }
            .dark-theme .save_pdf_button:disabled { background: #495057; }
            .landing-page-content-wrapper { min-height: 100vh; display: flex; justify-content: center; align-items: center; background: linear-gradient(270deg, #0d6efd, #6610f2, #0d6efd); background_size: 200% 200%; animation: gradient 15s ease infinite; padding: 20px; color: white; text-align: center; }
            .landing_page_content-wrapper h1, .landing-page-content-wrapper p { color: white; text_shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); }
            .landing-page-content-wrapper .btn { margin-top: 1.5rem; }
            body.rtl { direction: rtl; }
             @media (max-width: 767.98px) { .landing-page-content-wrapper .col-md-6 { text-align: center !important; align-items: center !important; }}
            @media (max-width: 576px) {
                .sidebar-header { font-size: 1.3rem; margin-bottom: 1rem;}
                .sidebar .nav-link { padding: 0.6rem 0.8rem; }
                .language-selector-wrapper { padding: 0 0.8rem; margin-bottom: 0.8rem;}
                .theme-switch-wrapper { padding: 0.4rem 0.8rem; margin-top: 0.8rem; }
                .button-wrapper { flex-direction: column; align-items: stretch; gap: 0.5rem; }
                .predict-button, .reset-button, .save_pdf_button { width: 100%; }
                #page-content { padding: 1rem; }
                .app-footer { font-size: 0.75rem; padding: 0.6rem 1rem;}
            }
        </style>
    </head>
    <body class="">
        <div id="app-title-container" style="display:none;"></div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            window.dash_clientside = window.dash_clientside || {};
            window.dash_clientside.clientside = {
                update_body_class_and_title: function(theme_value, lang_value, page_title, sidebar_state_data) {
                    const body = document.body;
                    const htmlElement = document.documentElement;
                    if (theme_value === 'dark') { body.classList.add('dark-theme'); }
                    else { body.classList.remove('dark-theme'); }
                    const isRTL = lang_value === 'ar';
                    if (isRTL) {
                        body.classList.add('rtl');
                        htmlElement.setAttribute('lang', 'ar');
                        htmlElement.setAttribute('dir', 'rtl');
                    } else {
                        body.classList.remove('rtl');
                        htmlElement.setAttribute('lang', lang_value || 'en');
                        htmlElement.setAttribute('dir', 'ltr');
                    }
                    if (page_title) { document.title = page_title; }
                    const isCollapsed = sidebar_state_data ? sidebar_state_data.collapsed : false;
                    const collapsedWidth = "''' + SIDEBAR_WIDTH_COLLAPSED + '''";
                    const expandedWidth = "''' + SIDEBAR_WIDTH_EXPANDED + '''";
                    const currentWidth = isCollapsed ? collapsedWidth : expandedWidth;
                    const pageContent = document.getElementById('page-content');
                    const footer = document.querySelector('.app-footer');
                    // Check if elements exist before trying to set styles
                    if (pageContent) {
                        if (isRTL) {
                            pageContent.style.marginRight = currentWidth;
                            pageContent.style.marginLeft = '0';
                        } else {
                            pageContent.style.marginLeft = currentWidth;
                            pageContent.style.marginRight = '0';
                        }
                    } else { console.warn("Page Content element (#page-content) not found."); } // pragma: no cover
                    if (footer) {
                         if (isRTL) {
                            footer.style.marginRight = currentWidth;
                            footer.style.marginLeft = '0';
                        } else {
                            footer.style.marginLeft = currentWidth;
                            footer.style.marginRight = '0';
                        }
                    } else { console.warn("Footer element not found."); } // pragma: no cover
                    return window.dash_clientside.no_update;
                },
                calculate_age_from_dob: function(dob_string) {
                    // Calculate age from DOB string (YYYY-MM-DD)
                    if (!dob_string) {
                        return window.dash_clientside.no_update; // No DOB, no update
                    }
                    try {
                        const birthDate = new Date(dob_string);
                        // Check if the date is valid after parsing
                        if (isNaN(birthDate.getTime())) {
                             console.warn("Invalid date from date picker:", dob_string); // Added logging
                            return window.dash_clientside.no_update;
                        }
                         // Adjust for timezone issues if necessary - common with datepickers
                         // Date picker often gives YYYY-MM-DD, which JS parses as UTC midnight
                         // Let's adjust it to be the start of the day in the user's local time
                         const userTimezoneOffset = birthDate.getTimezoneOffset() * 60000; // Offset in milliseconds
                         const adjustedBirthDate = new Date(birthDate.getTime() + userTimezoneOffset);


                        const today = new Date();
                         // Adjust today to start of day in local time for an accurate day-based comparison
                         const todayLocal = new Date(today.getFullYear(), today.getMonth(), today.getDate());


                        // Check if the adjusted birth date is in the future
                        if (adjustedBirthDate > todayLocal) {
                             // Allow edge case where DOB is exactly today (age 0)
                            if (adjustedBirthDate.getFullYear() === todayLocal.getFullYear() &&
                                adjustedBirthDate.getMonth() === todayLocal.getMonth() &&
                                adjustedBirthDate.getDate() === todayLocal.getDate()) {
                                return 0; // Age is 0 if born today
                            }
                             console.warn("DOB is in the future:", dob_string); // Added logging - FIX: Changed # to //
                            return window.dash_clientside.no_update; // DOB is clearly in the future, don't calculate age
                        }

                        // Calculate difference in years
                        let age = today.getFullYear() - adjustedBirthDate.getFullYear();
                        const m = today.getMonth() - adjustedBirthDate.getMonth();

                        // Adjust age if the birthday hasn't occurred yet this year
                        if (m < 0 || (m === 0 && today.getDate() < adjustedBirthDate.getDate())) {
                            age--;
                        }

                        // Check if the calculated age is within a reasonable range (e.g., 0-120)
                        if (age >= 0 && age <= 120) {
                            return age; // Return valid age
                        } else {
                             console.warn("Calculated age out of range (0-120):", age); // Added logging - FIX: Changed # to //
                            return window.dash_clientside.no_update; // pragma: no cover // Return no_update on any error during calculation
                        }
                    } catch (e) {
                         console.error("Error in calculate_age_from_dob:", e); // Added error logging - FIX: Changed # to //
                        return window.dash_clientside.no_update; // pragma: no cover // Return no_update on any error during calculation
                    }
                }
            };
        </script>
    </body>
</html>
'''

app.layout = html.Div(id="app-container", children=[
    # --- dcc.Store for session/local storage ---
    dcc.Store(id='theme-store', storage_type='local', data='light'), # Stores selected theme ('light' or 'dark')
    dcc.Store(id='language-store', storage_type='local', data=DEFAULT_LANG), # Stores selected language ('en', 'fr', 'ar')
    dcc.Store(id='sidebar-state-store', storage_type='local', data={'collapsed': False}), # Stores sidebar collapse state
    dcc.Location(id='url', refresh=False), # Represents the browser's address bar
    dcc.Store(id='current-page', data='landing'), # Tracks the current page being displayed

    # --- Invisible Div to trigger clientside title update ---
    html.Div(id='app-title-div', style={'display': 'none'}),

    # --- Dynamic Theme Stylesheet Link ---
    html.Link(id='theme-link', rel='stylesheet', href=light_theme_url),

    # --- dcc.Download component for PDF ---
    dcc.Download(id="download-pdf"),

    # --- Container for potential PDF error alerts ---
    html.Div(id='pdf-error-output'),

    # --- Main layout structure: sidebar + page content ---
    html.Div(id='main-content-wrapper', children=[
        html.Div(id='sidebar-content'), # Sidebar content (updated by callback)
        html.Div(id='page-content') # Main page content (updated by callback based on URL)
    ]),

    # --- Footer ---
    html.Footer(className="app-footer", children=[
        dbc.Container([dbc.Row([dbc.Col(html.P(id='footer-disclaimer-text', className="mb-0"))])], fluid=True)
    ]),

    # --- Dummy output for clientside callbacks that only trigger inputs ---
    html.Div(id='dummy-clientside-output', style={'display': 'none'})
])

# create_sidebar function (remains the same)
def create_sidebar(lang=DEFAULT_LANG, is_collapsed=False, current_theme='light'):
    sidebar_class = "sidebar collapsed" if is_collapsed else "sidebar"
    theme_is_dark = current_theme == 'dark'
    # Add a small margin-bottom to the last nav link before the language selector
    nav_links = [
        dbc.NavLink([html.I(className="fas fa-home fa-fw me-2"), html.Span(get_translation(lang, 'sidebar_home'), className="nav-text")], href="/", active="exact"),
        dbc.NavLink([html.I(className="fas fa-calculator fa-fw me-2"), html.Span(get_translation(lang, 'sidebar_predictor'), className="nav-text")], href="/dashboard", active="exact"),
        dbc.NavLink([html.I(className="fas fa-history fa-fw me-2"), html.Span(get_translation(lang, 'sidebar_history'), className="nav-text")], href="/history", active="exact"),
        # CORRECTED LINE BELOW: Use 'sidebar_about_model' key
        dbc.NavLink([html.I(className="fas fa-brain fa-fw me-2"), html.Span(get_translation(lang, 'sidebar_about_model'), className="nav-text")], href="/about-model", active="exact", className="mb-3"),
    ]

    return html.Div([
        html.Div([
            html.I(className="fas fa-laptop-medical me-2"),
            html.Span(get_translation(lang, 'sidebar_header'), className="header-text")
        ], className="sidebar-header"),
        html.Hr(style={'margin': '0.5rem 0'}),
        dbc.Nav(nav_links, vertical=True, pills=True, id="sidebar-nav"),
        html.Div([
            html.Label(get_translation(lang, 'select_language'), htmlFor="language-dropdown"),
            dcc.Dropdown(id='language-dropdown', options=[{'label': 'English', 'value': 'en'}, {'label': 'Français', 'value': 'fr'}, {'label': 'العربية', 'value': 'ar'}], value=lang, clearable=False, searchable=False),
        ], className="language-selector-wrapper"),
        html.Div([
            html.Label(get_translation(lang, 'theme_switch_label'), htmlFor="theme-switch"),
            dbc.Switch(id="theme-switch", value=theme_is_dark, className="ms-auto"),
        ], className="theme-switch-wrapper"),
        dbc.Button(html.I(id="sidebar-toggle-icon", className="fas fa-chevron-left" if not is_collapsed else "fas fa-chevron-right"), id="sidebar-toggle-button", className="sidebar-toggle-button", n_clicks=0),
    ], id="sidebar", className=sidebar_class)
# generate_pdf_report function (Corrected extraction of considerations)
def generate_pdf_report(lang, inputs, prediction_text, probability_text, considerations_list_of_strings):
    if not REPORTLAB_AVAILABLE: # pragma: no cover
        logger.error("Attempted to generate PDF, but ReportLab is not installed.")
        return None

    buffer = io.BytesIO() # Corrected: BytesBytes() -> BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []
    is_arabic_report_context = lang == 'ar'

    def get_display_text(plain_text_input, is_rtl_script_content_flag):
        if plain_text_input is None: return "" # Handle None input
        text_str = str(plain_text_input)
        # ReportLab needs specific font and bidi/reshape for complex scripts like Arabic
        if is_rtl_script_content_flag and REPORTLAB_ARABIC_TOOLS_AVAILABLE and _arabic_reshaper_module and _bidi_get_display_func: # pragma: no cover
            try:
                # Reshape Arabic characters and apply Bidi algorithm for correct display order
                reshaped_text = _arabic_reshaper_module.reshape(text_str)
                return _bidi_get_display_func(reshaped_text)
            except Exception as e: # pragma: no cover
                logger.warning(f"Could not reshape/bidi text for PDF ('{text_str[:30]}...'): {e}")
                return text_str # Fallback to raw text if reshaping fails
        return text_str

    def create_styled_paragraph(text_for_constructor,
                                base_style,
                                is_core_content_rtl,
                                report_is_rtl, # Controls default alignment
                                alignment_override=None,
                                is_numeric_block=False): # Hint for specific numeric alignment

        # Create a unique style name to avoid conflicts if called multiple times
        style_name = f"{base_style.name}_{len(story)}_{datetime.now().timestamp()}"
        current_style = ParagraphStyle(name=style_name, parent=base_style)
        # Ensure line spacing is reasonable (default 120% of font size)
        current_style.leading = getattr(base_style, 'leading', base_style.fontSize * 1.2 if hasattr(base_style, 'fontSize') else 12)

        # Apply Arabic font if needed and available
        if is_core_content_rtl and REPORTLAB_ARABIC_TOOLS_AVAILABLE: # pragma: no cover
            current_style.fontName = ARABIC_PDF_FONT_NAME

        # Determine text alignment
        if alignment_override is not None:
            current_style.alignment = alignment_override
        elif report_is_rtl: # pragma: no cover
             # Default to RIGHT alignment for RTL languages if no override
             current_style.alignment = TA_RIGHT
        else:
            # Use base style alignment (usually LEFT) for LTR
            current_style.alignment = base_style.alignment
            # Specific alignment for numeric blocks in LTR languages (often RIGHT)
            # This is handled better at the TableStyle level for consistent columns
            # Keeping this for non-table numeric paragraphs if needed.
            if is_numeric_block: # pragma: no cover
                 current_style.alignment = TA_RIGHT

        return Paragraph(text_for_constructor, current_style)

    title_style = styles['h1']
    title_style.fontSize = 16
    title_style.spaceAfter = 0.5*cm
    # Align title based on report language direction
    title_style.alignment = TA_RIGHT if is_arabic_report_context else TA_LEFT # Title aligned to edge
    # Center alignment override for the specific title Paragraph below

    header_style = styles['h3']
    header_style.spaceBefore = 0.5*cm
    header_style.spaceAfter = 0.2*cm
     # Align headers based on report language direction
    header_style.alignment = TA_RIGHT if is_arabic_report_context else TA_LEFT

    body_style = ParagraphStyle('BodyTextCustom', parent=styles['BodyText'])
    body_style.fontSize = 10
     # Align body text based on report language direction
    body_style.alignment = TA_RIGHT if is_arabic_report_context else TA_LEFT


    pdf_small_grey_style = ParagraphStyle('PdfSmallGrey', parent=styles['Normal'])
    pdf_small_grey_style.fontSize = 8
    pdf_small_grey_style.textColor = colors.grey
    pdf_small_grey_style.alignment = TA_RIGHT if is_arabic_report_context else TA_LEFT # Align based on report language
     # Center alignment override for the specific timestamp Paragraph below

    # Style for table headers - often bold or distinct
    # Base style inherited from Normal or BodyText for general properties
    header_cell_style_base = ParagraphStyle('HeaderCellBase', parent=styles['Normal'])
    header_cell_style_base.fontSize = 10
    header_cell_style_base.textColor = colors.black
    # Use a font that works for headers, potentially bold version of the body font
    # For Arabic, ensure it uses the Arabic font if needed
    header_cell_style_base.fontName = ARABIC_PDF_FONT_NAME if is_arabic_report_context and REPORTLAB_ARABIC_TOOLS_AVAILABLE else styles['h6'].fontName # Fallback to h6 font

    # --- Report Title ---
    raw_title_text = get_translation(lang, 'pdf_report_title')
    display_title_text = get_display_text(raw_title_text, is_arabic_report_context)
    story.append(create_styled_paragraph(
        display_title_text, title_style,
        is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context,
        alignment_override=TA_CENTER)) # Title is always centered

    # --- Generation Timestamp ---
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # The translation template might have the timestamp placeholder {timestamp}
    raw_generated_on_template = get_translation(lang, 'pdf_generated_on', timestamp='{TS_PLACEHOLDER}')
    # Construct the full string with actual timestamp before getting display text
    plain_semantic_text_generated_on = raw_generated_on_template.replace('{TS_PLACEHOLDER}', timestamp_str)
    display_generated_on_text = get_display_text(plain_semantic_text_generated_on, is_arabic_report_context)
    story.append(create_styled_paragraph(
        display_generated_on_text, pdf_small_grey_style,
        is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context,
        alignment_override=TA_CENTER)) # Timestamp is always centered
    story.append(Spacer(1, 0.5*cm))

    # --- Patient Data Header ---
    raw_patient_data_header = get_translation(lang, 'pdf_patient_data_header')
    display_patient_data_header = get_display_text(raw_patient_data_header, is_arabic_report_context)
    story.append(create_styled_paragraph(
        display_patient_data_header, header_style,
        is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context)) # Header alignment follows report direction

    # --- Patient Data Table ---
    table_headers_keys = ['pdf_parameter_header', 'pdf_value_header', 'pdf_unit_header']
    table_headers_paras = []
    for key in table_headers_keys:
        raw_text = get_translation(lang, key)
        # Process header text for display (Arabic shaping/bidi if needed)
        display_text = get_display_text(raw_text, is_arabic_report_context)
        # Create paragraph for table header cell - default to center
        # ReportLab's TableStyle ALIGN will handle actual column alignment
        table_headers_paras.append(create_styled_paragraph(
            display_text, header_cell_style_base,
            is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context,
            alignment_override=TA_CENTER)) # Table headers are centered content-wise

    input_data_for_pdf = [table_headers_paras]

    # Order of rows in the PDF table
    # Use the same internal keys as in the inputs_for_pdf_dict passed to this function
    # Include new inputs and ensure order matches headers
    pdf_input_keys_order = ['FirstName', 'LastName', 'DateOfBirth', 'Pregnancies', 'DiabetesPedigreeFunction', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
    # Corresponding translation keys for labels
    pdf_input_labels_order = ['pdf_first_name_header', 'pdf_last_name_header', 'pdf_dob_header', 'pdf_pregnancies_header', 'pdf_dpf_header', 'input_label_glucose', 'input_label_bp', 'input_label_skinthickness', 'input_label_insulin', 'input_label_bmi', 'input_label_age']

    for i, key in enumerate(pdf_input_keys_order):
        # Get the raw label text. For clinical params, get the name part before unit. For others, use full header key.
        if key in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']:
             raw_label_text_unprocessed = get_translation(lang, pdf_input_labels_order[i]).split('(')[0].strip()
        else:
             raw_label_text_unprocessed = get_translation(lang, pdf_input_labels_order[i]).strip()

        # Get the raw value data from the inputs dictionary
        value_data_raw = inputs.get(key, 'N/A') # 'N/A' -> key is missing

        # Determine the unit text. For core clinical params, get from CLINICAL_RANGES. For others (Name, DOB, Preg, DPF), it's empty.
        unit_text_raw = ""
        if key in CLINICAL_RANGES:
             unit_text_raw = CLINICAL_RANGES[key].get('unit', '')
             if key == 'Age': # Special handling for Age unit from translation
                full_age_label_for_pdf = get_translation(lang, 'input_label_age')
                if '(' in full_age_label_for_pdf and ')' in full_age_label_for_pdf: # pragma: no branch
                     unit_text_raw = full_age_label_for_pdf.split('(')[-1].split(')')[0].strip()
                # Fallback already handled by CLINICAL_RANGES lookup above

        # Determine if content needs RTL processing (label, value, unit)
        is_label_rtl_content = is_arabic_report_context
        # Value content RTL only for names/DOB if Arabic is the report language
        is_value_content_rtl_script = key in ['FirstName', 'LastName'] and is_arabic_report_context # pragma: no cover
        # Determine if the unit text itself needs Arabic processing (font and bidi/reshape)
        process_unit_as_arabic = is_arabic_report_context and bool(str(unit_text_raw).strip())

        # Format numeric values or handle non-numeric/None
        value_str_raw = 'N/A' # Default display string
        if value_data_raw is not None and str(value_data_raw).strip() != "":
            if isinstance(value_data_raw, (int, float, np.integer, np.floating)): # Check for various numeric types
                # Use appropriate formatting based on the parameter
                if key == 'BMI' or key == 'DiabetesPedigreeFunction': # DPF needs decimals
                    value_str_raw = f"{float(value_data_raw):.2f}" # DPF often has 2+ decimals
                elif key in ['Pregnancies', 'Age', 'BloodPressure', 'SkinThickness', 'Insulin', 'Glucose']: # Integers or 0 decimal for BP, ST, Insulin, Glucose (clinical context)
                    # Use float() first to handle inputs that might be string numbers
                    val_as_float = float(value_data_raw)
                    if key in ['Pregnancies', 'Age']:
                        value_str_raw = f"{int(val_as_float):.0f}" # Pregnancies and Age are integers
                    else: # Glucose, BloodPressure, SkinThickness, Insulin
                         # Check if the original input was likely an integer vs float for display preference
                        if val_as_float == int(val_as_float): # pragma: no cover
                             value_str_raw = f"{int(val_as_float):.0f}" # Display as integer if no decimal part
                        else:
                             value_str_raw = f"{val_as_float:.1f}" # Display with 1 decimal if there was a decimal part


            else: # Handle strings like FirstName, LastName, DateOfBirth
                 value_str_raw = str(value_data_raw)

        # Process text for display in PDF (shaping/bidi + font selection)
        display_label_text = get_display_text(raw_label_text_unprocessed, is_label_rtl_content)
        display_value_str = get_display_text(value_str_raw, is_value_content_rtl_script)
        display_unit_text = get_display_text(unit_text_raw, process_unit_as_arabic) # Apply Arabic processing to unit text if Arabic report

        # Create Paragraph objects for each cell, applying appropriate styles
        # Alignment for cell *content* is handled here, then TableStyle does *column* alignment
        # For RTL, content is naturally right-aligned if font/bidi is applied.
        # For LTR, align text/dates left, numbers right.

        para_label = create_styled_paragraph(
            display_label_text, body_style,
            is_core_content_rtl=is_label_rtl_content, report_is_rtl=is_arabic_report_context,
             alignment_override=TA_RIGHT if is_arabic_report_context else TA_LEFT) # Label alignment follows report direction

        # Determine value content type for content-level alignment
        is_value_content_numeric = key in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        value_para_alignment = TA_RIGHT if is_value_content_numeric and not is_arabic_report_context else (TA_RIGHT if is_arabic_report_context else TA_LEFT)
        # Arabic report: All content paragraphs should be TA_RIGHT internally for proper flow, regardless of numeric/text

        para_value = create_styled_paragraph(
            display_value_str, body_style,
            is_core_content_rtl=is_arabic_report_context, # Assume value content might need RTL processing in an Arabic report
            report_is_rtl=is_arabic_report_context,
            alignment_override=TA_RIGHT if is_arabic_report_context else (TA_RIGHT if is_value_content_numeric else TA_LEFT))

        para_unit = create_styled_paragraph(
            display_unit_text, body_style,
            is_core_content_rtl=process_unit_as_arabic, # Unit content requires RTL processing if Arabic unit in Arabic report
            report_is_rtl=is_arabic_report_context,
            alignment_override=TA_RIGHT if is_arabic_report_context else TA_LEFT) # Unit alignment follows report direction


        # Add the row to the table data
        input_data_for_pdf.append([para_label, para_value, para_unit])

    # Create the table object with specified column widths
    # Adjust widths slightly based on content/language if needed
    # Now have 3 columns + more rows, keep widths reasonable
    col_widths = [6*cm, 3*cm, 3*cm]


    input_table = Table(input_data_for_pdf, colWidths=col_widths)

    # Apply styles to the table (borders, background, padding, alignment)
    table_style_config = [
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), # Header row background
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),      # Header row text color
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke), # Body rows background (optional, can remove for cleaner look)
        ('GRID', (0,0), (-1,-1), 1, colors.black),        # All grid lines
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')             # Vertical alignment for all cells
    ]

    # Apply directional column alignment based on report language
    if is_arabic_report_context: # pragma: no cover
        # In RTL, align all columns to the RIGHT by default
        table_style_config.extend([
            ('ALIGN', (0,0), (-1,-1), 'RIGHT')
        ])
        # Note: Paragraph alignment *within* the cell content (handled by create_styled_paragraph)
        # takes precedence and is more important for complex scripts.

    else: # LTR (English/French)
        table_style_config.extend([
            # Specific column alignments for LTR
            ('ALIGN', (0,0), (0,-1), 'LEFT'),  # Parameter column (index 0) - Left
            ('ALIGN', (1,0), (1,-1), 'RIGHT'), # Value column (index 1) - Right (for numerics primarily)
            ('ALIGN', (2,0), (2,-1), 'LEFT')   # Unit column (index 2) - Left
        ])
        # Override default 'RIGHT' alignment in value column for non-numeric values (FirstName, LastName, DOB)
        # Add this only if the column was set to RIGHT by default (which it is above)
        for row_idx, row_data_key in enumerate(pdf_input_keys_order, start=1):
            # Keys that should be Left aligned in the value column (index 1) for LTR
            if row_data_key in ['FirstName', 'LastName', 'DateOfBirth']:
                 table_style_config.append(('ALIGN', (1, row_idx), (1, row_idx), 'LEFT'))


    input_table.setStyle(TableStyle(table_style_config))
    story.append(input_table)
    story.append(Spacer(1, 0.5*cm))

    # --- Prediction Result Header ---
    raw_prediction_header_text = get_translation(lang, 'pdf_prediction_header')
    display_prediction_header_text = get_display_text(raw_prediction_header_text, is_arabic_report_context)
    story.append(create_styled_paragraph(
        display_prediction_header_text, header_style,
        is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context)) # Header alignment follows report direction

    # --- Prediction Result Text ---
    raw_pred_text_plain = prediction_text if prediction_text and prediction_text != "N/A" else "N/A" #  "N/A" if no prediction text

    display_pred_text = get_display_text(raw_pred_text_plain, is_arabic_report_context)
    story.append(create_styled_paragraph(
        f"<b>{display_pred_text}</b>", body_style, # Make the result text bold using HTML tags
        is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context)) # Result text alignment follows report direction

    # --- Prediction Probability Text ---
    probability_text_for_pdf = str(probability_text) if probability_text is not None and isinstance(probability_text, str) else "N/A"
    probability_text_for_pdf = probability_text_for_pdf.strip()
    display_prob_text = get_display_text(probability_text_for_pdf, is_arabic_report_context)

    # Try to detect if the probability text contains a number (e.g., "Probability: 0.123")
    # For LTR, we might want to align the number part to the right.
    # For RTL, the whole string should be right-aligned anyway.
    # A simple approach for LTR is to check if the string looks like "Label: Number" and right-align the whole paragraph.
    is_prob_numeric_like = False
    if not is_arabic_report_context and ":" in probability_text_for_pdf: # pragma: no cover
        parts = probability_text_for_pdf.split(":", 1)
        if len(parts) == 2:
            prob_val_part = parts[1].strip()
            try:
                float(prob_val_part) # Check if the part after colon is numeric
                is_prob_numeric_like = True
            except ValueError: pass


    story.append(create_styled_paragraph(
        display_prob_text, body_style,
        is_core_content_rtl=is_arabic_report_context, # Probability string itself might need Bidi/reshaping in Arabic
        report_is_rtl=is_arabic_report_context,
        alignment_override=TA_RIGHT if is_arabic_report_context or is_prob_numeric_like else TA_LEFT
    ))
    story.append(Spacer(1, 0.5*cm))


    # --- Clinical Considerations Header ---
    raw_consid_header_text = get_translation(lang, 'pdf_considerations_header')
    display_consid_header_text = get_display_text(raw_consid_header_text, is_arabic_report_context)
    story.append(create_styled_paragraph(
        display_consid_header_text, header_style,
        is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context)) # Header alignment follows report direction

    # --- Clinical Considerations List/Text ---
    # considerations_list_of_strings now contains the extracted strings from Dash components.
    # It will either be the single "no considerations" message string or a list of markdown strings.

    # Check if the list is empty or contains only the "no considerations" message string
    no_consid_text_in_current_lang = get_translation(lang, 'pdf_no_considerations')
    is_empty_or_no_consid = not considerations_list_of_strings or (len(considerations_list_of_strings) == 1 and considerations_list_of_strings[0].strip() == no_consid_text_in_current_lang.strip()) # Use strip for robust comparison

    if not is_empty_or_no_consid:
        # Add the header *before* the list items if there are actual considerations
        # The header is already added above this section, but let's add the list context explicitly.
        # Maybe add a small intro paragraph if needed, but the header is likely enough.

        for consideration_markdown in considerations_list_of_strings:
            if not consideration_markdown or not consideration_markdown.strip(): continue # Skip empty strings

            # Process the raw markdown string:
            # 1. Get display text (reshaping/bidi for Arabic)
            # 2. Add a bullet point visually
            # 3. Convert **bold** markdown to <b>bold</b> HTML for ReportLab Paragraph
            # Ensure the order of operations: Bidi/Reshape on raw text -> Add bullet -> Convert markdown tags

            # Get the text processed for Bidi/Reshape
            display_text_content = get_display_text(consideration_markdown, is_arabic_report_context)

            # Add a visual bullet point. For RTL, put it at the end.
            bullet_char = "•"
            bulleted_display_text = f"{bullet_char} {display_text_content}" if not is_arabic_report_context else f"{display_text_content} {bullet_char}"

            # Convert markdown bold (**text**) to HTML bold (<b>text</b>) - simple conversion
            # This needs to be done AFTER getting the display text for Arabic, as markdown might break reshaping if not careful.
            # ReportLab's Paragraph parser handles HTML *after* the string is given, so reshaped+bidi text should be the input.
            # Let's try converting markdown *then* get display text - this is safer for Arabic flow.
            # Example: "**Glucose (150 mg/dL)**: Impaired tolerance."
            # Raw text: **Glucose (150 mg/dL)**: Impaired tolerance.
            # Markdown -> HTML: <b>Glucose (150 mg/dL)</b>: Impaired tolerance.
            # Get Display Text (on HTML string): Arabic font + Bidi/Reshape applied to the *content*, tags preserved.
            # Add bullet: • <b>Glucose (150 mg/dL)</b>: Impaired tolerance. (or reversed for RTL)

            # Let's try converting markdown first:
            html_text_content = consideration_markdown.replace('**', '<b>', 1).replace('**', '</b>', 1) # Converts the first pair
            # Now get display text for the HTML string
            display_html_text = get_display_text(html_text_content, is_arabic_report_context)
            # Add bullet
            bulleted_display_html_text = f"{bullet_char} {display_html_text}" if not is_arabic_report_context else f"{display_html_text} {bullet_char}"


            story.append(create_styled_paragraph(
                bulleted_display_html_text, body_style, # Pass the HTML string with bullet
                is_core_content_rtl=is_arabic_report_context, # The entire consideration text may need Bidi/reshaping
                report_is_rtl=is_arabic_report_context,
                alignment_override=TA_RIGHT if is_arabic_report_context else TA_LEFT)) # List item alignment follows report direction
            story.append(Spacer(1, 0.1*cm)) # Small space between list items
    else:
        # If no specific considerations were added, display the default "none" message
        raw_no_consid_text = get_translation(lang, 'pdf_no_considerations')
        display_no_consid_text = get_display_text(raw_no_consid_text, is_arabic_report_context)
        story.append(create_styled_paragraph(
            display_no_consid_text, body_style,
            is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context)) # Alignment follows report direction
    story.append(Spacer(1, 0.5*cm))

    # --- Legal Disclaimer ---
    raw_disclaimer_text = get_translation(lang, 'legal_disclaimer')
    display_disclaimer_text = get_display_text(raw_disclaimer_text, is_arabic_report_context)
    story.append(create_styled_paragraph(
        display_disclaimer_text, pdf_small_grey_style,
        is_core_content_rtl=is_arabic_report_context, report_is_rtl=is_arabic_report_context,
        alignment_override=TA_CENTER)) # Disclaimer is always centered

    # --- Build PDF ---
    try:
        # Attempt to build the PDF document
        doc.build(story)
        # Get the PDF content from the buffer
        pdf_data = buffer.getvalue()
        buffer.close() # Close the buffer
        logger.info("PDF report generated successfully.")
        return pdf_data
    except Exception as e: # pragma: no cover
        # Catch any errors during the PDF build process
        logger.error(f"Error building PDF with ReportLab: {e}", exc_info=True)
        if buffer: buffer.close() # Ensure buffer is closed even on error
        return None

# CLIENTSIDE CALLBACKS (allow_duplicate=True is needed on age Output)
clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='update_body_class_and_title'),
    Output('dummy-clientside-output', 'children'),
    Input('theme-store', 'data'), Input('language-store', 'data'),
    Input('app-title-div', 'children'), Input('sidebar-state-store', 'data')
)

clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='calculate_age_from_dob'),
    Output('age', 'value', allow_duplicate=True), # keep allow_duplicate=True here
    Input('date-of-birth', 'value'),
    prevent_initial_call=True
)

# SERVER-SIDE CALLBACKS
@app.callback(Output('app-title-div', 'children'), Input('language-store', 'data'))
def update_app_title(lang): return get_translation(lang, 'app_title')

@app.callback(
    Output('theme-link', 'href'), Output('theme-store', 'data'),
    Input('theme-switch', 'value'), State('theme-store', 'data'),
    prevent_initial_call=True
)
def update_theme(switch_value, current_theme_in_store):
    new_theme = 'dark' if switch_value else 'light'
    if new_theme == current_theme_in_store: return dash.no_update, dash.no_update # pragma: no cover
    return dark_theme_url if new_theme == 'dark' else light_theme_url, new_theme

@app.callback(
    Output('language-store', 'data'), Input('language-dropdown', 'value'),
    State('language-store', 'data'), prevent_initial_call=True
)
def update_language_store(selected_language, current_lang_in_store):
    if selected_language and selected_language in TRANSLATIONS and selected_language != current_lang_in_store:
        return selected_language
    return dash.no_update # pragma: no cover

@app.callback(
    Output('sidebar-state-store', 'data'), Input('sidebar-toggle-button', 'n_clicks'),
    State('sidebar-state-store', 'data'), prevent_initial_call=True
)
def toggle_sidebar_state(n_clicks, current_state):
    if n_clicks: return {'collapsed': not current_state.get('collapsed', False)}
    return dash.no_update # pragma: no cover

@app.callback(
    Output('sidebar-content', 'children'),
    Input('language-store', 'data'), Input('sidebar-state-store', 'data'), Input('theme-store', 'data')
)
def update_sidebar_content(lang, sidebar_state, theme):
    return create_sidebar(lang, sidebar_state.get('collapsed', False), theme)

@app.callback(Output('footer-disclaimer-text', 'children'), Input('language-store', 'data'))
def update_footer_disclaimer(lang): return get_translation(lang, 'legal_disclaimer')

# --- Define the landing page layout ---
#  --- Define the landing page layout as a function ---
def landing_page_layout(lang):
    # Check if model or scaler is available. If not, show an error landing page.
    # Include check for CatBoost Availability 

    if model_load_error: # pragma: no cover
        error_message_text = get_translation(lang, 'landing_error_message', error=model_load_error)
        error_suggestion_text = get_translation(lang, 'landing_error_suggestion')
        return dbc.Container([
            dbc.Row(dbc.Col([
                html.H1(get_translation(lang, 'landing_error_title'), className="display-4 mb-3 text-center"),
                html.P(error_message_text, className="lead text-center"),
                html.Hr(className="my-4"),
                html.P(error_suggestion_text, className="text-center"),
            ], md=8), justify="center", align="center", className="h-100")
        ], fluid=True, className="landing-page-content-wrapper") # Use the full height wrapper

    # If model loaded successfully, show the normal landing page
    return dbc.Container([
        dbc.Row(
            [
                dbc.Col([
                    html.H1(get_translation(lang, 'landing_title'), className="mb-3", style={"fontFamily": "Arial, sans-serif", "fontSize": "clamp(2.0rem, 5vw, 3.0rem)", "textShadow": "2px 2px 4px rgba(0, 0, 0, 0.3)", "animation": "slideIn 1.5s ease-out"}),
                    html.P(get_translation(lang, 'landing_subtitle'), className="mb-4", style={"fontSize": "clamp(1.0rem, 3vw, 1.4rem)", "fontWeight": "300", "animation": "slideIn 2s ease-out"}),
                    # Use dcc.Link with button inside for client-side navigation
                    dcc.Link(
                        dbc.Button(
                            get_translation(lang, 'landing_button'),
                            id='go-to-dashboard-button',
                            n_clicks=0,
                            color="light",
                            className="mt-4 px-5 py-3",
                            style={"fontSize": "clamp(1.0rem, 2.5vw, 1.2rem)", "borderRadius": "25px", "background": "linear-gradient(45deg, #ffffff, #e0e0e0)", "color": "#0d6efd", "border": "none", "animation": "pulse 2s infinite", "transition": "all 0.3s ease-in-out", "boxShadow": "0px 6px 12px rgba(0, 0, 0, 0.2)"}
                        ),
                        href="/dashboard" # Link to the dashboard page
                    )
                ], md=6, xs=12, className="d-flex flex-column justify-content-center align-items-center align-items-md-start text-center text-md-start"), # Added alignment classes
                dbc.Col(
                    html.Img(
                        src=app.get_asset_url('image.png'), 
                        alt=get_translation(lang, 'app_title'),
                        className="img-fluid",
                        style={'maxHeight': '70vh', 'maxWidth': '100%', 'objectFit': 'contain', 'animation': 'slideIn 1.8s ease-out'}
                    ),
                    md=6, xs=12, className="d-flex justify-content-center align-items-center mt-4 mt-md-0" # Added alignment classes and margin for small screens
                ),
            ],
            align="center", # Vertically align content in the row
            className="h-100" # Make row take full height of its container
        )
    ], fluid=True, className="landing-page-content-wrapper d-flex align-items-center") # Center container content

@app.callback(
    Output('page-content', 'children'), Output('current-page', 'data'),
    Input('url', 'pathname'),
    # Make language-store an Input to trigger page re-render on language change
    Input('language-store', 'data'),
    State('theme-store', 'data') # Keep theme as State, as theme change already triggers clientside body class
)
def display_page(pathname, lang_from_store, theme):
    
    lang = lang_from_store

    if lang not in TRANSLATIONS: lang = DEFAULT_LANG 
    if theme not in ['light', 'dark']: theme = 'light' 

    # --- History Page Layout ---
    history_page_layout = dbc.Container([
        dbc.Row([dbc.Col(html.H1(get_translation(lang, 'history_page_title'), className="mb-4 mt-2"), width=12)]),
        dbc.Row([dbc.Col(html.P(get_translation(lang, 'history_page_description')), width=12)]),
        dbc.Row([dbc.Col(dcc.Loading(id="loading-history-table", type="circle", children=[html.Div(id='history-table-content')]), width=12)]),
    ], fluid=True, className="mb-4")

    # --- About Model Page Layout ---
    about_model_page_layout = dbc.Container([
        dbc.Row([dbc.Col(html.H1(get_translation(lang, 'about_model_title'), className="mb-4 mt-2"), width=12)]),
        dbc.Row([dbc.Col([
            html.P(get_translation(lang, 'about_model_p1')),
            html.P(get_translation(lang, 'about_model_p2')),
            html.Hr(className="my-4"), # Separator
            html.H5(get_translation(lang, 'about_model_features_title'), className="mt-4 mb-2"),
            # This text is now updated to explain the feature set including hidden ones
            html.P(get_translation(lang, 'about_model_features_list')),
            # Corrected English phrase here
            html.P(get_translation(lang, 'about_model_data_note'), className="text-muted small"),
            html.Hr(className="my-4"), # Separator
            html.H5(get_translation(lang, 'about_model_performance_note_title'), className="mt-4 mb-2"),
            html.P(get_translation(lang, 'about_model_performance_note_text')),
            html.Hr(className="my-4"), # Separator
            html.P(get_translation(lang, 'about_model_disclaimer'), className="fst-italic text-muted small"),
        ], md=10, lg=8)]), # Constrain column width on larger screens
    ], fluid=True, className="mb-4") # Fluid container with bottom margin
    # --- End About Model Page Layout ---


    # create_info_parameter_item function 
    def create_input_group(input_id, label_key, tooltip_key_en=None, type='number', step=None, min_val=None, max_val=None, placeholder_key=None, current_lang_code=DEFAULT_LANG, **tooltip_kwargs_en):
        label_text = get_translation(current_lang_code, label_key)

        effective_tooltip_key = tooltip_key_en
        tooltip_kwargs_to_use = tooltip_kwargs_en

        # Map English tooltip key to language specific one if needed
        tooltip_key_map = {
            'en': {
                'input_tooltip_glucose': 'input_tooltip_glucose',
                'input_tooltip_bp': 'input_tooltip_bp',
                'input_tooltip_skinthickness': 'input_tooltip_skinthickness',
                'input_tooltip_insulin': 'input_tooltip_insulin',
                'input_tooltip_bmi': 'input_tooltip_bmi',
                'input_tooltip_age': 'input_tooltip_age',
                'input_tooltip_pregnancies': 'input_tooltip_pregnancies',
                'input_tooltip_dpf': 'input_tooltip_dpf',
            },
            'fr': {
                'input_tooltip_glucose': 'input_tooltip_glucose', 
                'input_tooltip_bp': 'input_tooltip_bp',
                'input_tooltip_skinthickness': 'input_tooltip_skinthickness_fr',
                'input_tooltip_insulin': 'input_tooltip_insulin_fr', 
                'input_tooltip_bmi': 'input_tooltip_bmi_fr', 
                'input_tooltip_age': 'input_tooltip_age', 
                'input_tooltip_pregnancies': 'input_tooltip_pregnancies', 
                'input_tooltip_dpf': 'input_tooltip_dpf', 
            },
             'ar': {
                'input_tooltip_glucose': 'input_tooltip_glucose',
                'input_tooltip_bp': 'input_tooltip_bp',
                'input_tooltip_skinthickness': 'input_tooltip_skinthickness_ar', 
                'input_tooltip_insulin': 'input_tooltip_insulin_ar', 
                'input_tooltip_bmi': 'input_tooltip_bmi_ar', 
                'input_tooltip_age': 'input_tooltip_age',
                'input_tooltip_pregnancies': 'input_tooltip_pregnancies', 
                'input_tooltip_dpf': 'input_tooltip_dpf', 
            }
        }
        # Use the mapped key for the current language, fallback to English tooltip key if not found
        effective_tooltip_key = tooltip_key_map.get(current_lang_code, tooltip_key_map[DEFAULT_LANG]).get(tooltip_key_en, tooltip_key_en)


        if effective_tooltip_key in ['info_modal_inputs_glucose', 'info_modal_inputs_bp', 'info_modal_inputs_age', 'info_modal_inputs_bmi', 'info_modal_inputs_skinthickness', 'info_modal_inputs_insulin', 'info_modal_inputs_pregnancies', 'info_modal_inputs_dpf']: # pragma: no cover
            if effective_tooltip_key == 'info_modal_inputs_glucose': tooltip_kwargs_to_use = {k: v for k, v in tooltip_kwargs_en.items() if k in ['normal_max', 'prediabetes_min', 'prediabetes_max', 'diabetes_threshold']}
            elif effective_tooltip_key == 'info_modal_inputs_bp': tooltip_kwargs_to_use = {k: v for k, v in tooltip_kwargs_en.items() if k in ['hypertension_threshold', 'hypotension_threshold']}
            elif effective_tooltip_key == 'info_modal_inputs_bmi': tooltip_kwargs_to_use = {k: v for k, v in tooltip_kwargs_en.items() if k in ['underweight_threshold', 'normal_min', 'normal_max', 'overweight_min', 'overweight_max', 'obesity_threshold']}
            elif effective_tooltip_key == 'info_modal_inputs_age': tooltip_kwargs_to_use = {k: v for k, v in tooltip_kwargs_en.items() if k in ['advanced_age_threshold']}
            elif effective_tooltip_key == 'info_modal_inputs_skinthickness': tooltip_kwargs_to_use = {k: v for k, v in tooltip_kwargs_en.items() if k in ['normal_threshold', 'elevated_threshold']}
            elif effective_tooltip_key == 'info_modal_inputs_insulin': tooltip_kwargs_to_use = {k: v for k, v in tooltip_kwargs_en.items() if k in ['min', 'max']}
             # New keys have no specific kwargs
            else: tooltip_kwargs_to_use = {}  # Default empty
        else: # For simple tooltip keys like input_tooltip_glucose etc.
             # Check if the French tooltip needs mmol conversion values
             if current_lang_code == 'fr' and effective_tooltip_key == 'input_tooltip_glucose':
                 tooltip_kwargs_to_use = tooltip_kwargs_en # Keep the kwargs for mmol conversion
             else:
                 tooltip_kwargs_to_use = {} # Simple tooltips don't use kwargs


        tooltip_text = get_translation(current_lang_code, effective_tooltip_key, **tooltip_kwargs_to_use) if effective_tooltip_key else None
        placeholder_text = get_translation(current_lang_code, placeholder_key) if placeholder_key else " "

        input_props = {"id": input_id, "type": type, "placeholder": placeholder_text, "className": "form-control"}
        if type == 'number':
            if min_val is not None: input_props["min"] = min_val
            if max_val is not None: input_props["max"] = max_val
            if step is not None: input_props["step"] = step
            # Add pattern for mobile keyboards if needed, but type=number handles a lot
        elif type == 'date':
            input_props["max"] = date.today().isoformat()
        elif type == 'text':
             pass # No specific props needed for text beyond placeholder

        children = [dbc.Input(**input_props), html.Label(label_text, htmlFor=input_id, id=f"{input_id}-label")]
        # Add tooltip if text exists and not date/text type (tooltips on labels are better for numeric inputs)
        if tooltip_text and type != 'date' and type != 'text':
             children.append(dbc.Tooltip(tooltip_text, target=f"{input_id}-label", placement="top"))
        children.append(html.Div(id=f'{input_id}-warning', className="warning-text")) # Area for validation warnings
        return html.Div(className="input-container", children=children)

    def create_info_parameter_item(param_key, lang_code, **kwargs):
        # Creates a ListGroupItem with formatted info for a single parameter
        full_text = get_translation(lang_code, param_key, **kwargs)
        title_part, description_part, example_part = "N/A", "", ""
        if ":" in full_text:
            parts = full_text.split(":", 1)
            title_part = parts[0].strip() + ":"
            remaining_text = parts[1].strip()
            # Identify example part based on language-specific keyword
            example_keyword = "Example:" if lang_code == 'en' else ("Exemple :" if lang_code == 'fr' else "مثال:")
            if example_keyword in remaining_text:
                desc_example_parts = remaining_text.split(example_keyword, 1)
                description_part = desc_example_parts[0].strip()
                example_part_raw = desc_example_parts[1].strip()
                example_part = f"{example_keyword.split(':')[0]}: {example_part_raw}" # Reconstruct example string without the colon in the label part
            else: # pragma: no cover
                description_part = remaining_text # No example found
        else: # pragma: no cover
            description_part = full_text # No colon found, treat whole text as description

        item_children = [
            html.H6(dcc.Markdown(f"**{title_part}**"), className="mb-1"), # Bold title using Markdown
            dcc.Markdown(description_part, className="mb-1 text-muted", style={'fontSize': '0.9rem'}) # Muted text for description
        ]
        if example_part:
            item_children.append(dcc.Markdown(f"*{example_part}*", className="text-info small fst-italic", style={'fontSize': '0.85rem'})) # Italic info text for example

        return dbc.ListGroupItem(item_children, className="border-0 px-0 py-2") # Borderless list item


    # Keyword arguments for info modal tooltips, using the clinical ranges defined earlier
    glucose_info_kwargs = CLINICAL_RANGES['Glucose']
    bp_info_kwargs = CLINICAL_RANGES['BloodPressure']
    skinthickness_info_kwargs = CLINICAL_RANGES['SkinThickness']
    insulin_info_kwargs = CLINICAL_RANGES['Insulin']
    bmi_info_kwargs = CLINICAL_RANGES['BMI']
    age_info_kwargs = {'advanced_age_threshold': CLINICAL_RANGES['Age'].get('advanced_age_threshold', 65)} # Only need advanced age for this info
    # No specific kwargs needed for Pregnancies or DPF info modal entries


    # Content for the info modal body (Accordion with parameter details)
    info_modal_accordion_body_content = dbc.ListGroup([
        create_info_parameter_item('info_modal_inputs_pregnancies', lang), # New
        create_info_parameter_item('info_modal_inputs_dpf', lang), # New
        create_info_parameter_item('info_modal_inputs_glucose', lang, **glucose_info_kwargs),
        create_info_parameter_item('info_modal_inputs_bp', lang, **bp_info_kwargs),
        create_info_parameter_item('info_modal_inputs_skinthickness', lang, **skinthickness_info_kwargs),
        create_info_parameter_item('info_modal_inputs_insulin', lang, **insulin_info_kwargs),
        create_info_parameter_item('info_modal_inputs_bmi', lang, **bmi_info_kwargs),
        create_info_parameter_item('info_modal_inputs_age', lang, **age_info_kwargs),
    ], flush=True) # Remove borders between list items

    info_modal_content = dbc.ModalBody([
        dbc.Accordion([
            dbc.AccordionItem([
                html.P(get_translation(lang, 'info_modal_accordion_inputs_p1'), className="mb-3"),
                info_modal_accordion_body_content
            ], title=get_translation(lang, 'info_modal_accordion_inputs_title')), # Accordion item title
        ], start_collapsed=False, flush=True, always_open=True) # Accordion settings
    ])

    # --- Dashboard Page Layout ---
    # Check again for model loading error before creating dashboard content
    if model_load_error or not CATBOOST_AVAILABLE: # pragma: no cover
         # If model/scaler couldn't load or CatBoost is missing, display an error message on the dashboard page too
         dashboard_content = dbc.Container([dbc.Alert(get_translation(lang, 'landing_error_message', error=model_load_error or "CatBoost library not found"), color="danger", className="m-4")], fluid=True, className="mb-4")
    else:
        # PDF button state and tooltip based on ReportLab availability and language support
        pdf_generally_possible = REPORTLAB_AVAILABLE
        initial_pdf_button_disabled_state = True # PDF button is initially disabled
        pdf_button_tooltip_text = get_translation(lang, 'save_pdf_button_text') # Default tooltip

        # Update tooltip if PDF features are disabled
        if not pdf_generally_possible: # pragma: no cover
            pdf_button_tooltip_text = get_translation(lang, 'alert_pdf_disabled')
        elif lang == 'ar' and not REPORTLAB_ARABIC_TOOLS_AVAILABLE: # pragma: no cover
            pdf_button_tooltip_text = get_translation(lang, 'alert_pdf_arabic_tools_missing')

        # Keyword arguments for tooltips, using the clinical ranges defined earlier
        # These are needed for the create_input_group calls
        glucose_tooltip_kwargs = CLINICAL_RANGES['Glucose']
        bp_tooltip_kwargs = CLINICAL_RANGES['BloodPressure']
        skinthickness_tooltip_kwargs = CLINICAL_RANGES['SkinThickness']
        insulin_tooltip_kwargs = CLINICAL_RANGES['Insulin']
        bmi_tooltip_kwargs = CLINICAL_RANGES['BMI']
        age_tooltip_kwargs = {'advanced_age_threshold': CLINICAL_RANGES['Age'].get('advanced_age_threshold', 65)} # Only need advanced age for this tooltip
        # No specific kwargs for Pregnancies or DPF tooltips


        dashboard_content = dbc.Container([
            # Info Modal Button
            dbc.Button(html.I(className="fas fa-info-circle"), id="open-info-modal", color="link", className="info-button"),
            # Info Modal
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle(get_translation(lang, 'info_modal_title'))),
                info_modal_content, # Content generated outside based on language
                dbc.ModalFooter(
                    dbc.Button(get_translation(lang, 'close_button'), id="close-info-modal", className="ms-auto", n_clicks=0)
                )
            ], id="info-modal", size="lg", scrollable=True, is_open=False), # Modal is initially closed

            # Page Title
            dbc.Row([dbc.Col(html.H1(get_translation(lang, 'dashboard_title'), className="mb-4 mt-2"), width=12)]),

            dbc.Row([
                # Input Card (Left Column)
                dbc.Col(lg=6, md=12, sm=12, xs=12, className="mb-4", children=[
                    dbc.Card([
                        dbc.CardHeader(get_translation(lang, 'card_header_input')),
                        dbc.CardBody([
                            # Row 1: Name
                            dbc.Row([
                                dbc.Col(md=6, children=[create_input_group('first-name', 'first_name_label', type='text', placeholder_key='first_name_placeholder', current_lang_code=lang)]),
                                dbc.Col(md=6, children=[create_input_group('last-name', 'last_name_label', type='text', placeholder_key='last_name_placeholder', current_lang_code=lang)]),
                            ]),
                            # Row 2: DOB and Age
                             dbc.Row([
                                dbc.Col(md=6, children=[create_input_group('date-of-birth', 'dob_label', type='date', placeholder_key='dob_placeholder', current_lang_code=lang)]),
                                dbc.Col(md=6, children=[create_input_group('age', 'input_label_age', 'input_tooltip_age', min_val=0, max_val=120, current_lang_code=lang, **age_tooltip_kwargs)]),
                            ]),
                            html.Hr(className="my-3"), # Separator
                            # Row 3: Pregnancies and DPF
                            dbc.Row([
                                dbc.Col(md=6, children=[create_input_group('pregnancies', 'input_label_pregnancies', 'input_tooltip_pregnancies', type='number', min_val=0, step=1, current_lang_code=lang)]),
                                dbc.Col(md=6, children=[create_input_group('diabetes-pedigree-function', 'input_label_dpf', 'input_tooltip_dpf', type='number', min_val=0, step=0.001, current_lang_code=lang)]),
                            ]),
                             # Row 4: (Glucose, BP)
                            dbc.Row([
                                dbc.Col(md=6, children=[create_input_group('glucose', 'input_label_glucose', 'input_tooltip_glucose', min_val=0, current_lang_code=lang, **glucose_tooltip_kwargs)]),
                                dbc.Col(md=6, children=[create_input_group('blood-pressure', 'input_label_bp', 'input_tooltip_bp', min_val=0, current_lang_code=lang, **bp_tooltip_kwargs)]),
                            ]),
                            # Row 5:  (Skin Thickness, Insulin)
                            dbc.Row([
                                dbc.Col(md=6, children=[create_input_group('skin-thickness', 'input_label_skinthickness', 'input_tooltip_skinthickness', min_val=0, current_lang_code=lang, **skinthickness_tooltip_kwargs)]),
                                dbc.Col(md=6, children=[create_input_group('insulin', 'input_label_insulin', 'input_tooltip_insulin', min_val=0, current_lang_code=lang, **insulin_tooltip_kwargs)]),
                            ]),
                             # Row 6: (BMI)
                            dbc.Row([
                                dbc.Col(md=6, children=[create_input_group('bmi', 'input_label_bmi', 'input_tooltip_bmi', step=0.1, min_val=0, current_lang_code=lang, **bmi_tooltip_kwargs)]),
                            ]),
                            # Buttons
                            html.Div(className="button-wrapper mt-3", children=[
                                dbc.Button(get_translation(lang, 'predict_button_text'), id='predict-button', n_clicks=0, className="predict-button", disabled=(model is None or scaler is None or not CATBOOST_AVAILABLE)),
                                dbc.Button(get_translation(lang, 'reset_button_text'), id='reset-button', color="secondary", outline=True, n_clicks=0, className="reset-button"),
                                # PDF Save Button (initially disabled)
                                dbc.Button(
                                    [html.I(className="fas fa-file-pdf me-2"), get_translation(lang, 'save_pdf_button_text')],
                                    id='save-pdf-button',
                                    n_clicks=0,
                                    className="save-pdf-button",
                                    disabled=initial_pdf_button_disabled_state, # Control disabled state via callback
                                    title=pdf_button_tooltip_text # Control tooltip via callback
                                )
                            ]),
                            # Alerts for model loading/dependency issues - visible only if errors occur
                            dbc.Alert(
                                get_translation(lang, 'landing_error_message', error=model_load_error),
                                color="danger",
                                is_open=(model_load_error is not None), # Show if specific loading error occurred
                                className="mt-3", duration=0 # Keep open indefinitely
                            ),
                            dbc.Alert( # pragma: no cover
                                get_translation(lang, 'alert_model_unavailable') + " (CatBoost library not found)",
                                color="danger",
                                is_open=(model is None and model_load_error is None and not CATBOOST_AVAILABLE), # Show if model is None AND no specific load error AND CatBoost missing
                                className="mt-3", duration=0
                            ),
                        ]),
                    ]),
                    # Clinical Considerations Card
                    dbc.Card([
                        dbc.CardHeader(html.H5([html.I(className="fas fa-stethoscope me-2"), get_translation(lang, 'card_header_clinical')])),
                        dbc.CardBody(id='clinical-considerations-section', children=[
                            # Placeholder text, updated by callback
                            html.P(get_translation(lang, 'clinical_considerations_placeholder'), className="text-muted")
                        ])
                    ])
                ]),
                # Output and Graph Card (Right Column)
                dbc.Col(lg=6, md=12, sm=12, xs=12, className="mb-4", children=[
                    # Prediction Output Card
                    dbc.Card([
                        dbc.CardHeader(get_translation(lang, 'card_header_output')),
                        dbc.CardBody(
                            # Loading state wrapper
                            dcc.Loading(
                                id="loading-prediction",
                                type="circle", 
                                children=[
                                    # Prediction Result (Alert) - Updated by predict_diabetes_clinical
                                    html.Div(id='prediction-output', className="result-container", style={'minHeight': '80px'}),
                                    # Prediction Probability Text - Updated by predict_diabetes_clinical
                                    html.Div(id='prediction-probability', className="mt-2 text-center", style={"fontSize": "1.0rem", "fontWeight": "500"})
                                ],
                                className="loading-state" 
                            )
                        )
                    ], style={"minHeight": "150px"}, className="mb-4"), # Ensure minimum height even when empty, bottom margin
                    # Input Visualization Graph Card
                    dbc.Card([
                        dbc.CardHeader(id='input-graph-title', children=get_translation(lang, 'card_header_graph')), # Graph title updated by callback
                        dbc.CardBody(
                             dcc.Graph(
                                id='input-vs-normal-graph',
                                figure=create_default_graph(lang, theme), # Initial empty graph, updated by callback
                                style={"height": "300px"},
                                config={'displayModeBar': False, 'responsive': True} # Disable Plotly mode bar, enable responsiveness
                            )
                        )
                    ])
                ]),
            ]),
        ], fluid=True, className="mb-4") # Fluid container with bottom margin


    # Determine which page layout to display based on URL pathname
    if pathname == '/dashboard': content_to_display, current_page_type = dashboard_content, "dashboard"
    elif pathname == '/history': content_to_display, current_page_type = history_page_layout, "history"
    elif pathname == '/about-model': content_to_display, current_page_type = about_model_page_layout, "about_model"
    # Call the landing_page_layout function instead of using an undefined variable
    elif pathname == '/': content_to_display, current_page_type = landing_page_layout(lang), "landing" # Use the landing_page_layout function call here
    else:
         # Handle unknown paths by redirecting to landing or showing a 404
         
         content_to_display, current_page_type = landing_page_layout(lang), "landing" # Default to landing page, use the function call
         
    return content_to_display, current_page_type


@app.callback(
    Output("info-modal", "is_open"),
    [Input("open-info-modal", "n_clicks"), Input("close-info-modal", "n_clicks")],
    State("info-modal", "is_open"), prevent_initial_call=True
)
def toggle_modal(n_open, n_close, is_open):
    # Callback to open/close the info modal
    ctx = callback_context
    # Check if the trigger was one of the modal buttons
    if not ctx.triggered: return is_open 
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id in ["open-info-modal", "close-info-modal"]:
        return not is_open # Toggle the state
    return is_open

# validate_inputs callback (remains largely the same, minor adjustments for age)
@app.callback(
    # Outputs are the children (warning text) for each input's warning Div
    # Added outputs for pregnancies and dpf warnings
    [Output('first-name-warning', 'children'), Output('last-name-warning', 'children'), Output('date-of-birth-warning', 'children'),
     Output('pregnancies-warning', 'children'), Output('diabetes-pedigree-function-warning', 'children'), 
     Output('glucose-warning', 'children'), Output('blood-pressure-warning', 'children'), Output('skin-thickness-warning', 'children'),
     Output('insulin-warning', 'children'), Output('bmi-warning', 'children'), Output('age-warning', 'children')],

    [Input('first-name', 'value'), Input('last-name', 'value'), Input('date-of-birth', 'value'),
     Input('pregnancies', 'value'), Input('diabetes-pedigree-function', 'value'), 
     Input('glucose', 'value'), Input('blood-pressure', 'value'), Input('skin-thickness', 'value'),
     Input('insulin', 'value'), Input('bmi', 'value'), Input('age', 'value')],
    # current language for warning messages
    State('language-store', 'data')
)
# Updated function signature to include new inputs
def validate_inputs(fname, lname, dob_str, pregnancies, dpf, glucose, bp, st, insulin, bmi, age_from_input_field, lang):
    # Provides instant visual validation feedback as the user types/selects
    if lang not in TRANSLATIONS: lang = DEFAULT_LANG 
 
    warnings = [""] * 11 

    # Map internal keys to their label translation keys for warning messages
    # Added new keys for pregnancies and dpf
    key_to_label_key = {
        "FirstName": 'first_name_label', "LastName": 'last_name_label', "DateOfBirth": 'dob_label',
        "Pregnancies": 'input_label_pregnancies', "DiabetesPedigreeFunction": 'input_label_dpf', 
        "Glucose": 'input_label_glucose', "BloodPressure": 'input_label_bp',
        "SkinThickness": 'input_label_skinthickness', "Insulin": 'input_label_insulin',
        "BMI": 'input_label_bmi', "Age": 'input_label_age'
    }
    # Map internal key to its index in the inputs and warnings list
    # Added new keys to the order
    input_key_order = ["FirstName", "LastName", "DateOfBirth", "Pregnancies", "DiabetesPedigreeFunction", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
    # Updated input values list
    input_values = [fname, lname, dob_str, pregnancies, dpf, glucose, bp, st, insulin, bmi, age_from_input_field]


    def get_warning_message(value_to_check, internal_key):
        # Helper to generate a warning message for a single input
        label_key = key_to_label_key.get(internal_key)
        # Get the clean name (before unit in parenthesis) - Note: New inputs don't have units in labels
        name = get_translation(lang, label_key).split(' (')[0] if label_key else internal_key

        # No warnings for empty or whitespace-only text fields (will be caught by predict callback as missing)
        if value_to_check is None or (isinstance(value_to_check, str) and str(value_to_check).strip() == "" ):
             # Special case for DOB: if empty, no format warning
             if internal_key == "DateOfBirth": return ""
           
             return ""

        # Specific validation for Date of Birth
        if internal_key == "DateOfBirth":
            if not isinstance(value_to_check, str): # Should be string from DatePicker
                 # This case is unlikely with a DatePicker but defensive
                 return get_translation(lang, 'warning_invalid_date_format') # pragma: no cover
            try:
                birth_dt = datetime.strptime(value_to_check, "%Y-%m-%d").date()
                if birth_dt > date.today():
                    return get_translation(lang, 'warning_future_dob')
            except ValueError:
                # This covers incorrect date formats like "2023-13-40" or non-date strings
                return get_translation(lang, 'warning_invalid_date_format')
            return "" # No warning if date is valid and not future

        # Validation for numeric fields
        # Added Pregnancies and DiabetesPedigreeFunction to this check
        if internal_key in ["Pregnancies", "DiabetesPedigreeFunction", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]:
            try:
                # Attempt to convert to float for numeric checks
                val = float(str(value_to_check))

                # Check for negative values (applicable to all numeric inputs)
                if val < 0:
                     return get_translation(lang, 'alert_negative_value', name=name)

                # Specific non-zero checks for Glucose, BloodPressure (as per preprocessing)
                # Validation for predict flags <=0, let's align warnings
                # Also Pregnancies should probably be non-negative integer, DPF non-negative float
                if internal_key in ["Glucose", "BloodPressure"] and val == 0:
                     return get_translation(lang, 'alert_value_must_be_positive', name=name)

                # Specific range/threshold checks
                if internal_key == "Age":
                    if not (0 <= val <= 120):
                        return get_translation(lang, 'warning_age_range')
                if internal_key == "BMI" and val < 10: # Lower bound for BMI validity
                    return get_translation(lang, 'warning_bmi_min', name=name)
                 # Pregnancies should be an integer
                if internal_key == "Pregnancies" and not val.is_integer():
                     # Use a specific warning or a general invalid format
                     return get_translation(lang, 'warning_invalid_format', name=name) # Reusing format warning for non-integers


                # Optional: Warning for unusually high values (can be noisy, keep conservative)
                # Use clinical ranges max values for a rough heuristic
                if internal_key in CLINICAL_RANGES:
                    ref = CLINICAL_RANGES[internal_key]
                    # Check against a multiple of a relevant upper bound if it exists
                    # Example: check against 3x diabetes threshold for glucose, 1.5x hypertension for bp, 4x normal for skin/insulin, 2x obesity for bmi
                    # Avoid checking if the threshold is 0 or None
                    if internal_key == "Glucose" and ref.get('diabetes_threshold') is not None and ref['diabetes_threshold'] > 0 and val > ref['diabetes_threshold'] * 3: # pragma: no cover
                         return get_translation(lang, 'warning_unusual_high', name=name, val=val)
                    elif internal_key == "BloodPressure" and ref.get('hypertension_threshold') is not None and ref['hypertension_threshold'] > 0 and val > ref['hypertension_threshold'] * 1.5: # pragma: no cover
                         return get_translation(lang, 'warning_unusual_high', name=name, val=val)
                    elif internal_key == "SkinThickness" and ref.get('elevated_threshold') is not None and ref['elevated_threshold'] > 0 and val > ref['elevated_threshold'] * 4: # pragma: no cover
                         return get_translation(lang, 'warning_unusual_high', name=name, val=val)
                    elif internal_key == "Insulin" and ref.get('max') is not None and ref['max'] > 0 and val > ref['max'] * 4: # pragma: no cover
                         return get_translation(lang, 'warning_unusual_high', name=name, val=val)
                    elif internal_key == "BMI" and ref.get('obesity_threshold') is not None and ref['obesity_threshold'] > 0 and val > ref['obesity_threshold'] * 2: # pragma: no cover
                         return get_translation(lang, 'warning_unusual_high', name=name, val=val)
                    

                return "" # No warning if numeric value is valid

            except (ValueError, TypeError):
                # Value cannot be converted to float
                return get_translation(lang, 'warning_invalid_format', name=name)

        # No validation needed for FirstName, LastName format beyond emptiness check above
        return ""


    # Apply validation to each input field and store the result in the warnings list
    for i, key in enumerate(input_key_order):
        warnings[i] = get_warning_message(input_values[i], key)

    
    authoritative_age_from_dob_check = None
    if dob_str: # pragma: no branch
         authoritative_age_from_dob_check = calculate_age(dob_str) # Recalculate age from DOB state

    # If DOB validation passed (no DOB warning at index 2) AND authoritative_age_from_dob_check is not None...
    if warnings[input_key_order.index("DateOfBirth")] == "" and authoritative_age_from_dob_check is not None: # pragma: no branch
         # ...then clear the Age input field's warning
         warnings[input_key_order.index("Age")] = ""


    # Return the updated warnings list (11 elements)
    return warnings


# save_to_log function (remains the same)
def save_to_log(inputs, prediction_label, probability):
   
    if not inputs: # pragma: no cover
        logger.error("Log attempt failed: No input data provided.")
        return
    try:
        log_data = {
            # Use raw inputs or calculated age/prediction details
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "FirstName": inputs.get('FirstName'), 
            "LastName": inputs.get('LastName'),   
            "DateOfBirth": inputs.get('DateOfBirth'), 
            # Added Pregnancies and DPF to log data
            "Pregnancies": inputs.get('Pregnancies'),
            "DiabetesPedigreeFunction": inputs.get('DiabetesPedigreeFunction'),
            "Glucose": inputs.get('Glucose'),    
            "BloodPressure": inputs.get('BloodPressure'), 
            "SkinThickness": inputs.get('SkinThickness'), 
            "Insulin": inputs.get('Insulin'),     
            "BMI": inputs.get('BMI'),            
            "Age": inputs.get('Age'), 
            "PredictedOutcome": prediction_label, # Store the internal English key
            "PredictedProbability": f"{probability:.3f}" if isinstance(probability, (float, np.floating)) else str(probability)
        }
        # Ensure columns are in the correct order for CSV header/rows
        # Added Pregnancies and DPF to the column list
        columns = ["Timestamp", "FirstName", "LastName", "DateOfBirth", "Pregnancies", "DiabetesPedigreeFunction", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age", "PredictedOutcome", "PredictedProbability"]

        file_exists = os.path.isfile(PREDICTIONS_CSV)
        df = pd.DataFrame([log_data], columns=columns) # Create a DataFrame for this single row
        # Append to CSV, writing header only if file didn't exist
        df.to_csv(PREDICTIONS_CSV, mode='a', header=not file_exists, index=False, encoding='utf-8')
        logger.info(f"Prediction logged to CSV: {log_data}")
    except Exception as e: # pragma: no cover
        logger.error(f"Error during log saving: {e}", exc_info=True)


# predict_diabetes_clinical callback - MODIFIED TO CONSTRUCT 12 FEATURES AND LOG ENGLISH KEYS
@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-probability', 'children'),
     Output('save-pdf-button', 'disabled'),
     Output('save-pdf-button', 'title')],
    Input('predict-button', 'n_clicks'),
    [State('first-name', 'value'), State('last-name', 'value'), State('date-of-birth', 'value'),
     State('pregnancies', 'value'), State('diabetes-pedigree-function', 'value'), # Added new States
     State('glucose', 'value'), State('blood-pressure', 'value'), State('skin-thickness', 'value'),
     State('insulin', 'value'), State('bmi', 'value'), State('age', 'value'),
     State('current-page', 'data'),
     State('language-store', 'data')] # Need language state to get display text
)
# Updated function signature to include new States
def predict_diabetes_clinical(n_clicks, fname, lname, dob_str, pregnancies_str, dpf_str, glucose_str, bp_str, st_str, insulin_str, bmi_str, age_input_val_str, current_page, lang):
    # This callback triggers the prediction and updates the output elements and PDF button state
    if lang not in TRANSLATIONS: lang = DEFAULT_LANG # pragma: no cover

    # Determine PDF button state and tooltip based on ReportLab availability and language support
    pdf_generally_possible = REPORTLAB_AVAILABLE
    pdf_possible_for_current_lang = pdf_generally_possible and (lang != 'ar' or REPORTLAB_ARABIC_TOOLS_AVAILABLE)
    default_pdf_button_title = get_translation(lang, 'save_pdf_button_text')
    current_pdf_button_title = default_pdf_button_title
    if not pdf_generally_possible: # pragma: no cover
        current_pdf_button_title = get_translation(lang, 'alert_pdf_disabled')
    elif lang == 'ar' and not REPORTLAB_ARABIC_TOOLS_AVAILABLE: # pragma: no cover
        current_pdf_button_title = get_translation(lang, 'alert_pdf_arabic_tools_missing')

    # Initial states for outputs (before prediction runs or on error)
    initial_pred_output = ""
    initial_prob_output = ""
    initial_pdf_disabled_state = True # PDF button is disabled by default (until successful prediction)

    # Check if model or scaler is available. If not, return error message.
    # This check should include CATBOOST_AVAILABLE since the model file requires it.
    if model is None or scaler is None or not CATBOOST_AVAILABLE: # pragma: no cover
        error_msg = get_translation(lang, 'alert_model_unavailable')
        if model_load_error:
             # If a specific loading error occurred (e.g., file not found), show that
             error_msg = get_translation(lang, 'landing_error_message', error=model_load_error)
        elif not CATBOOST_AVAILABLE: # pragma: no cover
             # If CatBoost is just not installed
             error_msg = get_translation(lang, 'alert_model_unavailable') + " (CatBoost library not found)"

        return (dbc.Alert(error_msg, color="danger"),
                initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title)


    triggered_id = callback_context.triggered_id
    # Only run the prediction logic if the 'predict-button' was clicked on the 'dashboard' page
    if not triggered_id or triggered_id != 'predict-button' or n_clicks is None or n_clicks == 0 or current_page != 'dashboard':
        # If not triggered correctly, return initial outputs
        return initial_pred_output, initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title

    # --- Data Collection and Validation ---

    # Determine the authoritative Age value (prefer DOB calculation)
    authoritative_age_value = None
    if dob_str:
        calculated_age_from_dob = calculate_age(dob_str)
        if calculated_age_from_dob is not None:
            authoritative_age_value = calculated_age_from_dob
    # If DOB is not available or invalid, use the value from the Age input field if valid
    if authoritative_age_value is None: # pragma: no cover
        if age_input_val_str is not None and str(age_input_val_str).strip() != "":
            try:
                authoritative_age_value = int(float(str(age_input_val_str)))
                # Re-validate age range here just in case clientside failed or input directly
                if not (0 <= authoritative_age_value <= 120):
                     authoritative_age_value = None # Treat as invalid if outside range
            except (ValueError, TypeError): # pragma: no cover
                authoritative_age_value = None
        else:
            authoritative_age_value = None # Age remains None if neither source provides a valid age

    # Map of raw/collected inputs from the UI fields
    # Use authoritative age for Age
    # Added Pregnancies and DPF to the raw inputs map
    raw_inputs_map = {
        "FirstName": fname, "LastName": lname, "DateOfBirth": dob_str,
        "Pregnancies": pregnancies_str, "DiabetesPedigreeFunction": dpf_str, # Added new inputs
        "Glucose": glucose_str, "BloodPressure": bp_str, "SkinThickness": st_str,
        "Insulin": insulin_str, "BMI": bmi_str, "Age": authoritative_age_value
    }

  
    app_input_keys_numeric_required = ["Pregnancies", "DiabetesPedigreeFunction", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # List of translation keys for the input labels, used for error messages
   
    label_keys_order = ['first_name_label', 'last_name_label', 'dob_label', 'input_label_pregnancies', 'input_label_dpf', 'input_label_glucose', 'input_label_bp', 'input_label_skinthickness', 'input_label_insulin', 'input_label_bmi', 'input_label_age']
    # Create a mapping from internal key (like "Glucose") to its index in the label_keys list
    
    input_key_order_all = ["FirstName", "LastName", "DateOfBirth", "Pregnancies", "DiabetesPedigreeFunction", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
    key_to_label_index = {k:i for i, k in enumerate(input_key_order_all)}


    # Dictionary to hold cleaned and validated numeric values for the 8 collected features
    # These values will be used for feature engineering
    cleaned_numeric_inputs = {}
    # Dictionary to hold raw/cleaned values for logging purposes
    # Use the correct uppercase keys here to match the CSV column headers
    input_dict_for_log = {}

    # Add non-numeric/special fields to the log dictionary first using their raw values
    input_dict_for_log['FirstName'] = raw_inputs_map.get('FirstName') 
    input_dict_for_log['LastName'] = raw_inputs_map.get('LastName')   
    input_dict_for_log['DateOfBirth'] = raw_inputs_map.get('DateOfBirth') 

    # Handle Age validation separately as it can come from DOB or input field
    age_val = authoritative_age_value
    input_dict_for_log['Age'] = age_val 

    # Age is now checked alongside other required numeric inputs below

    # Iterate through the required numeric base inputs collected by the app (including age now)
    # Include Age in this loop for consistent validation and adding to cleaned_numeric_inputs
    app_input_keys_numeric_required_including_age = app_input_keys_numeric_required + ['Age']

    for internal_key_current in app_input_keys_numeric_required_including_age:
        # Get the raw value from the inputs map
        raw_value = raw_inputs_map[internal_key_current]
        # Get the translated UI name for this input
        label_key_for_ui = label_keys_order[key_to_label_index[internal_key_current]]
        ui_name = get_translation(lang, label_key_for_ui).split(' (')[0] # Split by '(' for clinical names

        # Handle specific unit cases for UI name lookup if needed, but splitting should work

        # Add the raw value to the log dictionary (will be overwritten by cleaned value if needed)
        input_dict_for_log[internal_key_current] = raw_value

        # --- Server-Side Validation for Prediction ---
        # Check if required numeric input is missing or empty
        if raw_value is None or (isinstance(raw_value, str) and str(raw_value).strip() == "" ):
            
             if internal_key_current in ["Pregnancies", "DiabetesPedigreeFunction"]:
                 # Log as N/A or None, but continue to use placeholder below
                 input_dict_for_log[internal_key_current] = None # Log it explicitly as None/missing
                 # Use .get with default 0.0 in case the key is somehow missing from PLACEHOLDER_MEANS
                 cleaned_numeric_inputs[internal_key_current] = PLACEHOLDER_MEANS.get(internal_key_current, 0.0)
                 logger.info(f"Missing input for '{internal_key_current}', using placeholder mean: {cleaned_numeric_inputs[internal_key_current]}")
                 continue # Skip further validation for this input if just missing

             # For ALL other inputs (Glucose, BP, ST, Insulin, BMI, Age), missing is a fatal error
             return (dbc.Alert(get_translation(lang, 'alert_missing_value', name=ui_name), color="warning"),
                     initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title)


        try:
            # Convert the value to float for validation and feature engineering
            val_f = float(str(raw_value))

            # Perform validation based on known ranges/constraints
            if internal_key_current in ["Glucose", "BloodPressure"] and val_f <= 0:
                 return dbc.Alert(get_translation(lang, 'alert_value_must_be_positive', name=ui_name), color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title
            if internal_key_current in ["SkinThickness", "Insulin", "Pregnancies", "DiabetesPedigreeFunction"] and val_f < 0:
                 return dbc.Alert(get_translation(lang, 'alert_negative_value', name=ui_name), color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title
            if internal_key_current == "BMI" and val_f < 10: # Arbitrary lower bound for BMI
                  return dbc.Alert(get_translation(lang, 'alert_low_bmi'), color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title
            if internal_key_current == "Age" and not (0 <= val_f <= 120): # Age range validation
                  return dbc.Alert(get_translation(lang, 'alert_invalid_age'), color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title
            # Pregnancies must be an integer
            if internal_key_current == "Pregnancies" and not val_f.is_integer():
                 return dbc.Alert(get_translation(lang, 'warning_invalid_format', name=ui_name), color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title


            # Store the successfully validated and converted numeric value for feature engineering
            cleaned_numeric_inputs[internal_key_current] = val_f
            # Update the log dict with the cleaned numeric value
            input_dict_for_log[internal_key_current] = val_f


        except (ValueError, TypeError):
            # Return an error if the value is not a valid number
            return dbc.Alert(get_translation(lang, 'alert_invalid_numeric', name=ui_name), color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title

    # --- Feature Engineering for the Scaler/Model (Construct the 12 features) ---

    # Get the cleaned numeric values for feature engineering. These should exist after validation
    # or be set to placeholder means for Pregnancies/DPF if input was missing.
    pregnancies_val = cleaned_numeric_inputs.get('Pregnancies')
    glucose_val = cleaned_numeric_inputs.get('Glucose')
    bp_val = cleaned_numeric_inputs.get('BloodPressure')
    st_val = cleaned_numeric_inputs.get('SkinThickness')
    insulin_val = cleaned_numeric_inputs.get('Insulin')
    bmi_val = cleaned_numeric_inputs.get('BMI')
    dpf_val = cleaned_numeric_inputs.get('DiabetesPedigreeFunction') # Correctly get DPF
    age_val = cleaned_numeric_inputs.get('Age') # This is the authoritative age


    # Calculate engineered features, precisely matching preprocessing logic (including +1 denominators)
    
    glucose_bmi_ratio_calc = glucose_val / (bmi_val + 1.0) 
    age_bmi_calc = age_val * bmi_val
    insulin_glucose_ratio_calc = insulin_val / (glucose_val + 1.0) 
    pregnancies_age_ratio_calc = pregnancies_val / (age_val + 1.0) 


    # Create a dictionary mapping ALL 12 expected feature names to their calculated or placeholder values for THIS patient.
    # This dictionary must contain *all* keys in FEATURE_ORDER_FOR_SCALER.
    feature_values_map = {
        # Use values from cleaned_numeric_inputs (which include user input or placeholder mean)
        'Pregnancies': pregnancies_val,
        'Glucose': glucose_val,
        'BloodPressure': bp_val,
        'SkinThickness': st_val,
        'Insulin': insulin_val,
        'BMI': bmi_val,
        'DiabetesPedigreeFunction': dpf_val,
        'Age': age_val,
        # Engineered Features - use the calculated values
        'Glucose_BMI_Ratio': glucose_bmi_ratio_calc,
        'Age_BMI': age_bmi_calc,
        'Insulin_Glucose_Ratio': insulin_glucose_ratio_calc,
        'Pregnancies_Age_Ratio': pregnancies_age_ratio_calc 
    }

    # Construct the final 12-element list of values in the EXACT required order
    # Iterate through the predefined correct order and get values from the map
    features_list_ordered = []
    try:
        features_list_ordered = [feature_values_map[col_name] for col_name in FEATURE_ORDER_FOR_SCALER]
        # logger.info(f"Constructed ordered features list: {features_list_ordered}") # Uncomment for debugging
    except KeyError as e: # pragma: no cover
         # This indicates a mismatch between FEATURE_ORDER_FOR_SCALER and the keys in feature_values_map
         # i.e., a feature name expected by the scaler is not being generated.
         logger.error(f"Internal feature mismatch: Expected feature '{e}' not found in generated feature map.", exc_info=True)
         return dbc.Alert(get_translation(lang, 'alert_prediction_error') + f" (Internal feature setup error: {e})", color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title


    try:
        # Convert the 12-element list into a NumPy array of shape (1, 12) for the scaler
        features_array = np.array([features_list_ordered])
       
        scaled_features = scaler.transform(features_array)
       
        prediction_proba_val = model.predict_proba(scaled_features)[0]
        # predict returns the predicted class label (0 or 1)
        prediction_val = model.predict(scaled_features)[0]
        # --- End CatBoost Prediction ---

        # --- Prepare Prediction Output ---
        result_text_key, result_color, icon_class, prob_text_display, prediction_label_for_log, prob_for_log = "", "", "", "", "", 0.0

        # Determine the output message based on the predicted class
        # Use the internal English keys for logging
        if prediction_val == 0: # Assuming 0 is the 'negative' class (no diabetes)
            result_text_key = 'pred_result_low_risk'
            result_color = "success"
            icon_class = "fas fa-check-circle"
            # Display probability of the predicted class (class 0 for low risk)
            prob_text_display = get_translation(lang, 'pred_prob_low_risk', prob=prediction_proba_val[0])
            prediction_label_for_log = get_translation('en', 'pred_result_low_risk') # Store English key in log
            prob_for_log = prediction_proba_val[0] # Log probability of class 0
        else: # Assuming 1 is the 'positive' class (diabetes)
            result_text_key = 'pred_result_high_risk'
            result_color = "danger"
            icon_class = "fas fa-exclamation-triangle"
            # Display probability of the predicted class (class 1 for high risk)
            prob_text_display = get_translation(lang, 'pred_prob_high_risk', prob=prediction_proba_val[1])
            prediction_label_for_log = get_translation('en', 'pred_result_high_risk') # Store English key in log
            prob_for_log = prediction_proba_val[1] # Log probability of class 1


        # Log the prediction result with patient inputs
        save_to_log(input_dict_for_log, prediction_label_for_log, prob_for_log)

        # Create the Dash output components (Alert for result, Div for probability)
        result_display_alert = dbc.Alert([
            html.I(className=f"{icon_class} me-2"), # Icon next to text
            html.Strong(get_translation(lang, result_text_key)) # Bold result text
        ], color=result_color, className='d-flex align-items-center justify-content-center mb-0') # Center align content in the alert

        # Determine if the PDF button should be enabled.
        # It should be enabled only if PDF features are possible AND a prediction was successfully made.
        # Prediction is successful if we reached this point without returning an error Alert.
        final_pdf_disabled_state = not pdf_possible_for_current_lang

        return result_display_alert, prob_text_display, final_pdf_disabled_state, current_pdf_button_title

    except Exception as e: # pragma: no cover
        # Catch any unexpected errors during the scaling, prediction, or output formatting process
        logger.error(f"Prediction error: {e}", exc_info=True)
        # Return a generic error message alert
        return dbc.Alert(get_translation(lang, 'alert_prediction_error'), color="danger"), initial_prob_output, initial_pdf_disabled_state, current_pdf_button_title


# update_clinical_considerations callback (Corrected typo CLINICAL_RAGES -> CLINICAL_RANGES)
@app.callback(
    Output('clinical-considerations-section', 'children'), Input('predict-button', 'n_clicks'),
    [State('glucose', 'value'), State('blood-pressure', 'value'), State('skin-thickness', 'value'),
     State('insulin', 'value'), State('bmi', 'value'),
     State('date-of-birth', 'value'), State('age', 'value'), # Added age input state
     # Added new states for Pregnancies and DPF - These are NOT used for *clinical ranges* but collected as part of *inputs*
     State('pregnancies', 'value'), State('diabetes-pedigree-function', 'value'),
     State('current-page', 'data'), State('language-store', 'data')]
)
# Updated function signature to include new inputs (though they won't trigger specific clinical notes based on ranges)
def update_clinical_considerations(n_clicks, glucose, bp, st, insulin, bmi, dob_str, age_input_val, pregnancies, dpf, current_page, lang):
    # This callback generates clinical notes based *only* on raw input values relative to thresholds
    # It does NOT depend on the model prediction or the 12-feature vector

    # Only run if the predict button was clicked on the dashboard page
    # Or on initial load if figure is None and there's data to display? No, keep it simple, tied to predict click.
    if not callback_context.triggered_id or callback_context.triggered_id != 'predict-button' or n_clicks is None or n_clicks == 0 or current_page != 'dashboard':
        # Return the initial placeholder text
        return html.P(get_translation(lang, 'clinical_considerations_placeholder'), className="text-muted")

    if lang not in TRANSLATIONS: lang = DEFAULT_LANG # pragma: no cover

    # Determine the authoritative Age value (prefer DOB calculation)
    authoritative_age_value = None
    if dob_str: # pragma: no branch
        calculated_age_from_dob = calculate_age(dob_str)
        if calculated_age_from_dob is not None: # pragma: no branch
            authoritative_age_value = calculated_age_from_dob
    # If DOB is not available or invalid, use the value from the Age input field if valid
    if authoritative_age_value is None and age_input_val is not None and str(age_input_val).strip() != "": # pragma: no cover
        try: authoritative_age_value = int(float(str(age_input_val)))
        except (ValueError, TypeError): authoritative_age_value = None


    # Map of inputs relevant for clinical considerations
    # Include Age using the authoritative value
    # Note: Pregnancies and DPF are in inputs_map but won't generate considerations based on ranges below
    inputs_map = {
        "Glucose": {'value': glucose, 'label_key': 'input_label_glucose'},
        "BloodPressure": {'value': bp, 'label_key': 'input_label_bp'},
        "SkinThickness": {'value': st, 'label_key': 'input_label_skinthickness'},
        "Insulin": {'value': insulin, 'label_key': 'input_label_insulin'},
        "BMI": {'value': bmi, 'label_key': 'input_label_bmi'},
        "Age": {'value': authoritative_age_value, 'label_key': 'input_label_age'}, # Use authoritative age
        "Pregnancies": {'value': pregnancies, 'label_key': 'input_label_pregnancies'}, # Included in map but no clinical range logic below
        "DiabetesPedigreeFunction": {'value': dpf, 'label_key': 'input_label_dpf'} # Included in map but no clinical range logic below
    }

    considerations_output_components = [] # List to store generated consideration HTML elements
    considerations_output_strings = [] # List to store generated consideration text strings (for PDF extraction)


    # Helper function to safely convert input value to float, returns None on failure or empty string
    def safe_float(v):
        if v is None or (isinstance(v, str) and v.strip() == ""): return None
        try: return float(v)
        except (ValueError, TypeError): return None


    # Check if any relevant input values were provided for considerations (focusing on those with clinical ranges)
    has_any_relevant_input = any(safe_float(data['value']) is not None for key, data in inputs_map.items() if key in CLINICAL_RANGES)
    if not has_any_relevant_input:
        # Return placeholder if no valid clinical inputs are present for ranged parameters
        # Add the "no considerations" message to the strings list as well
        no_consid_text = get_translation(lang, 'clinical_considerations_none')
        considerations_output_strings.append(no_consid_text)
        return html.P(no_consid_text, className="text-muted") # Show the 'none' message if no relevant inputs

    # Iterate through each relevant input to generate considerations
    for internal_key, data in inputs_map.items():
        # Only process keys for which we have clinical ranges defined
        if internal_key not in CLINICAL_RANGES:
            continue # Skip if no clinical reference defined (This excludes Pregnancies and DPF)

        val_num = safe_float(data['value']) # Get the numeric value

        # If value is None or invalid, skip consideration for this parameter
        if val_num is None:
            continue

        # Get clinical reference data for this parameter
        clinical_ref = CLINICAL_RANGES[internal_key]
        # Get the clean label name
        ui_name = get_translation(lang, data['label_key']).split(' (')[0]
        # Get the unit for display
        unit_text = clinical_ref.get('unit', '')
        if internal_key == 'Age': # Special handling for Age unit from translation
            full_age_label = get_translation(lang, 'input_label_age')
            if '(' in full_age_label and ')' in full_age_label: # pragma: no branch
                unit_text = full_age_label.split('(')[-1].split(')')[0].strip()
            elif 'unit' in CLINICAL_RANGES['Age']: # Fallback
                 # Corrected typo here from CLINICAL_RAGES to CLINICAL_RANGES
                 unit_text = CLINICAL_RANGES['Age']['unit']
            else: # pragma: no cover
                 unit_text = ""


        consideration_text = "" # Initialize consideration text string
        li_html_class = "normal" # Default class for the list item (color coding)


        # Generate consideration text based on value thresholds for each parameter
        if internal_key == "Glucose":
            val_mmol = val_num / 18.0182 # Convert mg/dL to mmol/L for potential display (FR translation uses this)
            if val_num >= clinical_ref.get('diabetes_threshold', 200): # >= 200
                consideration_text = get_translation(lang, 'consideration_glucose_diabetes', name=ui_name, val=val_num, unit=unit_text, val_mmol=val_mmol)
                li_html_class = "highlight" # Red color
            elif val_num >= clinical_ref.get('prediabetes_min', 140): # 140-199
                consideration_text = get_translation(lang, 'consideration_glucose_prediabetes', name=ui_name, val=val_num, unit=unit_text, val_mmol=val_mmol)
                li_html_class = "moderate-alert" # Amber color
            elif val_num >= 0 and val_num <= clinical_ref.get('normal_max', 139): # 0-139
                consideration_text = get_translation(lang, 'consideration_glucose_normal', name=ui_name, val=val_num, unit=unit_text, val_mmol=val_mmol)
                li_html_class = "normal" # Green color
        elif internal_key == "BloodPressure":
             # Use get with default values
            if val_num > clinical_ref.get('hypertension_threshold', 90):
                consideration_text = get_translation(lang, 'consideration_bp_hypertensive', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "moderate-alert" # Amber color
            elif val_num < clinical_ref.get('hypotension_threshold', 60) and val_num >= 0: # Ensure non-negative and below hypotension
                consideration_text = get_translation(lang, 'consideration_bp_hypotensive', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "moderate-alert" # Amber color
            elif val_num >= clinical_ref.get('hypotension_threshold', 60) and val_num <= clinical_ref.get('hypertension_threshold', 90): # Within normal/borderline range
                consideration_text = get_translation(lang, 'consideration_bp_normal', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "normal" # Green color
        elif internal_key == "SkinThickness":
             # Use get with default to handle potential missing keys in CLINICAL_RANGES if structure changes
            if val_num > clinical_ref.get('elevated_threshold', 23): # Example threshold 23mm
                consideration_text = get_translation(lang, 'consideration_skinthickness_elevated_female', name=ui_name, val=val_num, unit=unit_text, threshold=clinical_ref.get('elevated_threshold', 23))
                li_html_class = "moderate-alert" # Amber color
            elif val_num >= 0: # Consider 0+ as potentially normal or low
                consideration_text = get_translation(lang, 'consideration_skinthickness_normal_female', name=ui_name, val=val_num, unit=unit_text, threshold=clinical_ref.get('normal_threshold', 23))
                li_html_class = "normal" # Green color
        elif internal_key == "Insulin":
             # Use get with default
            if val_num > clinical_ref.get('max', 166): # Example high threshold 166 mUI/L
                consideration_text = get_translation(lang, 'consideration_insulin_high', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "moderate-alert" # Amber color
            elif val_num < clinical_ref.get('min', 16) and val_num >= 0: # Example low threshold 16 mUI/L, and non-negative
                 consideration_text = get_translation(lang, 'consideration_insulin_low', name=ui_name, val=val_num, unit=unit_text)
                 li_html_class = "moderate-alert" # Amber color
            elif val_num >= clinical_ref.get('min', 16) and val_num <= clinical_ref.get('max', 166): # Within normal range
                consideration_text = get_translation(lang, 'consideration_insulin_normal', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "normal" # Green color
        elif internal_key == "BMI":
             # Use get with default
            if val_num >= clinical_ref.get('obesity_threshold', 30): # >= 30
                consideration_text = get_translation(lang, 'consideration_bmi_obesity', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "highlight" # Red color
            elif val_num >= clinical_ref.get('overweight_min', 25): # 25-29.9
                consideration_text = get_translation(lang, 'consideration_bmi_overweight', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "moderate-alert" # Amber color
            elif val_num >= clinical_ref.get('normal_min', 18.5): # 18.5-24.9
                consideration_text = get_translation(lang, 'consideration_bmi_normal', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "normal" # Green color
            elif val_num >= 10 and val_num < clinical_ref.get('underweight_threshold', 18.5): # < 18.5 (assuming min BMI 10 as per validation)
                consideration_text = get_translation(lang, 'consideration_bmi_underweight', name=ui_name, val=val_num, unit=unit_text)
                li_html_class = "moderate-alert" 

        elif internal_key == "Age":
            age_unit_for_trans = unit_text # Use the unit already determined
            if val_num >= clinical_ref.get('advanced_age_threshold', 65): # >= 65
                consideration_text = get_translation(lang, 'consideration_age_advanced', name=ui_name, val=val_num, unit_age=age_unit_for_trans)
                li_html_class = "moderate-alert" # Amber color (Age is a non-modifiable risk)
            elif val_num >= 0 and val_num < clinical_ref.get('young_adult_max', 35) : # < 35 and non-negative
                consideration_text = get_translation(lang, 'consideration_age_young', name=ui_name, val=val_num, unit_age=age_unit_for_trans)
                li_html_class = "normal" # Green color
            elif val_num >= 0: # Adult age (35-64) and non-negative
                consideration_text = get_translation(lang, 'consideration_age_adult', name=ui_name, val=val_num, unit_age=age_unit_for_trans)
                li_html_class = "normal" # Green color


        # If a consideration text was generated, add it to both lists
        if consideration_text:
            # For the UI component: Wrap in List Item with class and dcc.Markdown
            considerations_output_components.append(html.Li(dcc.Markdown(consideration_text, className="mb-0", dangerously_allow_html=False), className=li_html_class))
            # For the PDF string list: Add the raw markdown string
            considerations_output_strings.append(consideration_text)


    # If no specific considerations were added but inputs were present (i.e., all inputs were within "normal" ranges),
    # the `considerations_output_components` list might still be empty if no specific "normal" consideration text was defined for that parameter.
    # In this case, show the "none" message.
    # Also, if inputs were present but none triggered a specific text, ensure the strings list gets the "none" message for PDF.
    if not considerations_output_components:
        no_consid_text = get_translation(lang, 'clinical_considerations_none')
        considerations_output_strings.append(no_consid_text) 
        return html.P(no_consid_text, className="text-muted")

    # Return the list of considerations wrapped in a Div and a header for the UI
    # Also store the raw text strings in a hidden div for later retrieval by the PDF callback
    return html.Div([
        html.P(get_translation(lang, 'clinical_considerations_header'), className="mb-2 fw-bold"), # Header text
        html.Ul(considerations_output_components, className="clinical-considerations-list"), # List of considerations UI components
        html.Div(id='clinical-considerations-text-for-pdf', style={'display': 'none'}, children=considerations_output_strings) # Hidden div to store strings for PDF
    ])


# update_input_vs_normal_graph callback (remains the same)
@app.callback(
    [Output('input-vs-normal-graph', 'figure'), Output('input-graph-title', 'children')],
    Input('predict-button', 'n_clicks'),
    [State('glucose', 'value'), State('blood-pressure', 'value'), State('skin-thickness', 'value'),
     State('insulin', 'value'), State('bmi', 'value'),
     State('date-of-birth', 'value'), State('age', 'value'), # Added age input state
     # Added new states for Pregnancies and DPF - these are NOT added to the graph
     State('pregnancies', 'value'), State('diabetes-pedigree-function', 'value'),
     State('theme-store', 'data'), State('language-store', 'data'), State('current-page', 'data')]
)
# Updated function signature to include new inputs (though they are not used by the graph logic)
def update_input_vs_normal_graph(n_clicks, glucose_str, bp_str, st_str, insulin_str, bmi_str, dob_str, age_input_val_str, pregnancies_str, dpf_str, current_theme, lang, current_page):
    # Updates the graph visualization of patient values vs clinical reference ranges
    if lang not in TRANSLATIONS: lang = DEFAULT_LANG
    graph_header_text = get_translation(lang, 'card_header_graph') # Get translated graph header

    # Only run if the predict button was clicked on the dashboard page
    if current_page != 'dashboard' or not callback_context.triggered_id or callback_context.triggered_id != 'predict-button' or n_clicks is None or n_clicks == 0:
        # Return the initial empty graph and header text
        return create_default_graph(lang, current_theme), graph_header_text

    # Determine the authoritative Age value (prefer DOB calculation)
    authoritative_age_value = None
    if dob_str:
        calculated_age = calculate_age(dob_str)
        if calculated_age is not None: authoritative_age_value = calculated_age
    # If DOB age is not available/valid, use the age from the input field if valid
    if authoritative_age_value is None and age_input_val_str is not None and str(age_input_val_str).strip() != "": # pragma: no cover
        try: authoritative_age_value = int(float(str(age_input_val_str)))
        except (ValueError, TypeError): authoritative_age_value = None

    # Determine Plotly template based on theme
    plotly_template = "plotly_dark" if current_theme == 'dark' else "plotly_white"
    is_dark_theme = (current_theme == 'dark')

    # Map inputs to their string values and label keys for processing
    # Only include inputs relevant for the graph (those with clinical ranges defined)
    inputs_data_map = {
        "Glucose": {'value_str': glucose_str, 'label_key': 'input_label_glucose'},
        "BloodPressure": {'value_str': bp_str, 'label_key': 'input_label_bp'},
        "SkinThickness": {'value_str': st_str, 'label_key': 'input_label_skinthickness'},
        "Insulin": {'value_str': insulin_str, 'label_key': 'input_label_insulin'},
        "BMI": {'value_str': bmi_str, 'label_key': 'input_label_bmi'},
        # Use authoritative age (converted back to string for safe_float check below)
        "Age": {'value_str': str(authoritative_age_value) if authoritative_age_value is not None else None, 'label_key': 'input_label_age'}
        # Pregnancies and DPF are NOT included in this map for the graph
    }

    # Lists to hold data for plotting
    categories_for_graph = [] # X-axis labels (e.g., "Glucose", "BMI")
    patient_values_for_graph = [] # Y-values for patient markers
    # Data for the filled reference range polygons (uses Scatter trace with fill='toself')
    x_coords_for_fill = []
    y_coords_lower_for_fill = [] # Y-coordinates for the bottom edge of the rectangle

    units_for_hover_list = [] # Units for hover text on patient markers
    patient_value_colors = [] # Colors for patient markers
    # Reference range min/max for hover text on patient markers
    hover_ref_min_list = []
    hover_ref_max_list = []

    has_valid_data_for_graph = False # Flag to check if we have any data to plot
    current_x_index = 0 # Counter for positioning categories on the x-axis

    # Process each input parameter that has a defined clinical range
    # Iterate in a consistent order for the graph display
    graph_order_keys = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]

    for internal_key in graph_order_keys:
        # Get the data for the current key from the map
        data = inputs_data_map.get(internal_key)
        if data is None: continue # Should not happen with the list above, but defensive

        value_str_current = data['value_str']
        label_translation_key = data['label_key']
        # Get the clean UI name for the graph label
        ui_name_for_graph = get_translation(lang, label_translation_key).split(' (')[0]

        # Safely convert the input string value to a number
        try:
            val_numeric = float(value_str_current) if value_str_current is not None and str(value_str_current).strip() != "" else None
        except (ValueError, TypeError):
            val_numeric = None

        # Only include this parameter in the graph if it has a valid numeric value AND a defined clinical range
        if val_numeric is not None and internal_key in CLINICAL_RANGES:
            has_valid_data_for_graph = True # We have at least one data point
            clinical_ref = CLINICAL_RANGES[internal_key] # Get clinical ranges

            # Get the unit for hover text
            unit_str = clinical_ref.get('unit', 'N/A')
            if internal_key == 'Age': # Special handling for Age unit
                full_age_label = get_translation(lang, 'input_label_age')
                if '(' in full_age_label and ')' in full_age_label: # pragma: no branch
                    unit_str = full_age_label.split('(')[-1].split(')')[0].strip()
                elif 'unit' in CLINICAL_RANGES['Age']: # Fallback
                     unit_str = CLINICAL_RANGES['Age']['unit']
                else: # pragma: no cover
                     unit_str = "N/A"


            categories_for_graph.append(ui_name_for_graph) # Add label to X-axis categories
            patient_values_for_graph.append(val_numeric)   # Add patient value for marker Y-position
            units_for_hover_list.append(unit_str)          # Add unit for hover text

            # --- Define Data for Reference Range Polygon ---
            # These min/max define the shaded rectangular area for this parameter based on graph_ref
            ref_min_for_fill = clinical_ref.get("graph_ref_min")
            ref_max_for_fill = clinical_ref.get("graph_ref_max")

            if ref_min_for_fill is not None and ref_max_for_fill is not None: # pragma: no branch
                 # Add coordinates for a rectangle shape [bottom-left, bottom-right, top-right, top-left, close_path]
                 # X-coordinates are relative to the current categorical position (current_x_index)
                 x_coords_for_fill.extend([current_x_index - 0.4, current_x_index + 0.4, current_x_index + 0.4, current_x_index - 0.4, None]) # None to separate shapes
                 y_coords_lower_for_fill.extend([ref_min_for_fill, ref_min_for_fill, ref_max_for_fill, ref_max_for_fill, None])


            # --- Define Data for Hover Text Reference Range ---
            # These min/max are shown in the tooltip for the patient marker.
            # Use clinical 'normal' or relevant thresholds here, not necessarily the graph_ref ones.
            hr_min, hr_max = None, None # Default to None if no specific hover range
            if internal_key == "Glucose": hr_min, hr_max = 0, clinical_ref.get('normal_max', 139) # 0 to Normal Max
            elif internal_key == "BloodPressure": hr_min, hr_max = clinical_ref.get('hypotension_threshold', 60), clinical_ref.get('hypertension_threshold', 90) -1 # Hypotension to Hypertension-1
            elif internal_key == "SkinThickness": hr_min, hr_max = 0, clinical_ref.get('normal_threshold', 23) # 0 to Normal Max
            elif internal_key == "Insulin": hr_min, hr_max = clinical_ref.get('min', 16), clinical_ref.get('max', 166) # Min to Max
            elif internal_key == "BMI": hr_min, hr_max = clinical_ref.get('normal_min', 18.5), clinical_ref.get('normal_max', 24.9) # Normal Weight Range
            elif internal_key == "Age": hr_min, hr_max = clinical_ref.get('hover_ref_min',20), clinical_ref.get('hover_ref_max',65) # Typical Adult Age Range

            hover_ref_min_list.append(hr_min)
            hover_ref_max_list.append(hr_max)

            # --- Determine Patient Marker Color based on Value ---
            marker_color = "#6c757d" # Default grey color
            if internal_key == "Glucose":
                if val_numeric >= clinical_ref.get('diabetes_threshold', 200): marker_color = "#dc3545" # Red for Diabetes
                elif val_numeric >= clinical_ref.get('prediabetes_min', 140): marker_color = "#ffc107" # Amber for Prediabetes
                elif val_numeric >= 0 and val_numeric <= clinical_ref.get('normal_max', 139): marker_color = "#198754" # Green for Normal
            elif internal_key == "BloodPressure":
                if val_numeric > clinical_ref.get('hypertension_threshold', 90) or (val_numeric < clinical_ref.get('hypotension_threshold', 60) and val_numeric >= 0): marker_color = "#ffc107" # Amber for abnormal BP (and non-negative)
                elif val_numeric >= clinical_ref.get('hypotension_threshold', 60) and val_numeric <= clinical_ref.get('hypertension_threshold', 90): marker_color = "#198754" # Green for Normal/Borderline
            elif internal_key == "SkinThickness":
                if val_numeric > clinical_ref.get('elevated_threshold', 23): marker_color = "#ffc107" # Amber for Elevated
                elif val_numeric >= 0: marker_color = "#198754" # Green for Normal/Low
            elif internal_key == "Insulin":
                if val_numeric > clinical_ref.get('max',166) or (val_numeric < clinical_ref.get('min',16) and val_numeric >= 0): marker_color = "#ffc107" # Amber for High or Low 
                elif val_numeric >= clinical_ref.get('min',16) and val_numeric <= clinical_ref.get('max',166): marker_color = "#198754" # Green for Normal
            elif internal_key == "BMI":
                if val_numeric >= clinical_ref.get('obesity_threshold',30): marker_color = "#dc3545" # Red for Obesity
                elif val_numeric >= clinical_ref.get('overweight_min',25): marker_color = "#ffc107" # Amber for Overweight
                elif val_numeric >= clinical_ref.get('normal_min',18.5): marker_color = "#198754" # Green for Normal Weight
                elif val_numeric >= 10 and val_numeric < clinical_ref.get('underweight_threshold', 18.5): marker_color = "#ffc107" # Amber for Underweight
            elif internal_key == "Age":
                if val_numeric >= clinical_ref.get('advanced_age_threshold', 65) : marker_color = "#ffc107" # Amber for Advanced Age
                elif val_numeric >= 0 and val_numeric < clinical_ref.get('young_adult_max', 35) : marker_color = "#198754" # Green for Young Adult 
                elif val_numeric >= 0: marker_color = "#198754" # Green for Adult 

            # Adjust colors for dark theme visibility
            if is_dark_theme: 
                if marker_color=="#198754": marker_color="#4ade80" # Brighter Green
                elif marker_color=="#ffc107": marker_color="#facc15" # Brighter Amber
                elif marker_color=="#dc3545": marker_color="#f87171" # Brighter Red
                elif marker_color=="#6c757d": marker_color="#adb5bd" # Lighter Grey

            patient_value_colors.append(marker_color) # Add color for this marker
            current_x_index += 1 # Increment index for the next parameter


    # If no valid data points were found for any parameter, return the default empty graph
    if not has_valid_data_for_graph or not categories_for_graph:
        return create_default_graph(lang, current_theme), graph_header_text

    # --- Create Plotly Figure ---
    fig = go.Figure()

    # Define color for the shaded reference range area based on theme
    ref_fill_color_light = 'rgba(220,220,220,0.5)' # Light grey for light theme
    ref_fill_color_dark = 'rgba(80,80,80,0.5)'   # Darker grey for dark theme
    ref_fill_color = ref_fill_color_dark if is_dark_theme else ref_fill_color_light

    # Add the filled reference range trace
    if x_coords_for_fill and y_coords_lower_for_fill: # Ensure there are coordinates to plot
        fig.add_trace(go.Scatter(
            x=x_coords_for_fill,
            y=y_coords_lower_for_fill,
            fill='toself', # This creates a closed shape for each group of 4 points (rectangle)
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'), # Make the line invisible
            fillcolor=ref_fill_color,
            hoverinfo='skip', # Don't show hover for the fill area itself
            name=get_translation(lang,'graph_legend_ref'), # Legend text for reference range
            legendgroup='ref_range', # Group traces for the legend
            showlegend=True # Ensure this trace appears in the legend
        ))
    else: # pragma: no cover
         # Add a dummy trace for the legend item if no data for fill (e.g., if all inputs are for params without ranges)
         fig.add_trace(go.Scatter(
             x=[None], y=[None], mode='markers', marker=dict(color=ref_fill_color, size=10, symbol='square'),
             name=get_translation(lang,'graph_legend_ref'), legendgroup='ref_range', showlegend=True
         ))


    # Define outline color for patient markers based on theme
    marker_outline_color = 'rgba(60,60,60,0.8)' if is_dark_theme else 'rgba(200,200,200,0.8)'

    # Prepare custom data for the patient markers' hover text
    # It needs to be a NumPy array where each row corresponds to a patient marker (parameter)
    # Columns should be [hover_ref_min, hover_ref_max, unit]
    # Ensure all lists have the same length as categories_for_graph
    if not (len(hover_ref_min_list) == len(hover_ref_max_list) == len(units_for_hover_list) == len(categories_for_graph) == len(patient_values_for_graph) == len(patient_value_colors)):
        logger.error("Mismatch in lengths of lists for graph customdata. Creating default graph.") # pragma: no cover
        return create_default_graph(lang, current_theme), graph_header_text # pragma: no cover

    custom_data_for_hover = np.stack((hover_ref_min_list, hover_ref_max_list, units_for_hover_list), axis=-1)
    # Get the hover template string based on the current language
    hover_template_string = get_translation(lang, 'graph_hover_template')

    # X-values for the patient markers. These are simply indices 0, 1, 2, ... corresponding to categories_for_graph.
    marker_x_values = list(range(len(categories_for_graph)))

    # Add the patient value markers trace
    fig.add_trace(go.Scatter(
        name=get_translation(lang,'graph_legend_patient'), # Legend text for patient value
        x=marker_x_values, # Use numerical indices for x
        y=patient_values_for_graph, # Use patient's values for y
        mode='markers', # Only show markers
        marker=dict(
            color=patient_value_colors, # Use determined colors
            size=13,
            symbol='circle',
            line=dict(width=1, color=marker_outline_color) # Add a subtle outline
        ),
        customdata=custom_data_for_hover, # Attach custom data for hover text
        hovertemplate=hover_template_string, # Use the custom hover template
        legendgroup='patient_values', # Group traces for the legend
        showlegend=True
    ))

    # --- Determine Y-axis Range ---
    # Find the overall min and max of all plotted values (patient values and reference ranges)
    all_y_values_for_range_calc = []
    if y_coords_lower_for_fill: # pragma: no branch
        # Extract valid numeric values from the fill coordinates
        # Need to handle the 'None' separators in y_coords_lower_for_fill
        valid_fill_y = [y for y in y_coords_lower_for_fill if y is not None and isinstance(y, (int, float, np.number))]
        all_y_values_for_range_calc.extend(valid_fill_y)

    if patient_values_for_graph: # pragma: no branch
        # Extract valid numeric values from patient values
        valid_patient_y = [y for y in patient_values_for_graph if y is not None and isinstance(y, (int, float, np.number))]
        all_y_values_for_range_calc.extend(valid_patient_y)

    min_y_val_overall = 0 # Start min range at 0 (clinical values are typically non-negative)
    max_y_val_overall = 100 # Default max range if no data or only zero data

    if all_y_values_for_range_calc: # Check if the list is not empty after filtering
        min_y_val_overall = min(all_y_values_for_range_calc)
        max_y_val_overall = max(all_y_values_for_range_calc)

        # Handle edge case where all values are the same
        if min_y_val_overall == max_y_val_overall: # pragma: no cover
             # Add some padding if all values are identical
             padding_amount = abs(min_y_val_overall * 0.2) if min_y_val_overall != 0 else 10 # 20% or a fixed 10 if value is 0
             max_y_val_overall = min_y_val_overall + padding_amount
             # Ensure min doesn't go below zero if original values were zero
             min_y_val_overall = max(0, min_y_val_overall - (abs(min_y_val_overall * 0.2) if min_y_val_overall != 0 else 0))


    # Add some padding to the calculated range for better visualization
    y_axis_padding = (max_y_val_overall - min_y_val_overall) * 0.10 # 10% padding
    final_y_axis_min = max(0, min_y_val_overall - y_axis_padding) # Ensure min doesn't go below 0
    final_y_axis_max = max_y_val_overall + y_axis_padding

    # Prevent inverted range if max somehow ends up less than min (shouldn't happen with logic above, but defensive)
    if final_y_axis_min >= final_y_axis_max: # pragma: no cover
        final_y_axis_max = final_y_axis_min + 10 # Add a default height if range is flat or inverted

    # --- Update Layout ---
    fig.update_layout(
        title=get_translation(lang,'graph_title'), # Graph title based on language
        yaxis_title=get_translation(lang,'graph_yaxis_label'), # Y-axis label
        # Position legend horizontally at the top right
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=40), # Adjust margins
        template=plotly_template, # Apply theme template
        yaxis=dict(
            range=[final_y_axis_min, final_y_axis_max], # Set dynamic Y-axis range
            showgrid=True, # Show horizontal grid lines
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.1)' # Light grey grid lines
        ),
        xaxis=dict(
            tickmode = 'array', # Use array mode for categorical labels
            tickvals = marker_x_values, # Numerical positions (0, 1, 2...)
            ticktext = categories_for_graph, # Text labels (Glucose, BMI...)
            showgrid=False, # Hide vertical grid lines between categories
            # Position the x-axis label "Value" correctly if added
            # title_standoff=10 # May need adjustment based on label length
        ),
        paper_bgcolor='rgba(0,0,0,0)', # Make plot background transparent
        plot_bgcolor='rgba(0,0,0,0)'   # Make plot area background transparent
    )

    # Return the figure and the graph header text
    return fig, graph_header_text


# reset_form_and_results callback (Corrected: allow_duplicate=True on age Output)
@app.callback(
    [Output('first-name', 'value'), Output('last-name', 'value'), Output('date-of-birth', 'value'),
     Output('pregnancies', 'value'), Output('diabetes-pedigree-function', 'value'), # Added outputs for new inputs
     Output('glucose', 'value'), Output('blood-pressure', 'value'), Output('skin-thickness', 'value'),
     Output('insulin', 'value'), Output('bmi', 'value'), Output('age', 'value', allow_duplicate=True), # Added allow_duplicate=True here
     Output('first-name-warning', 'children', allow_duplicate=True), Output('last-name-warning', 'children', allow_duplicate=True), Output('date-of-birth-warning', 'children', allow_duplicate=True),
     Output('pregnancies-warning', 'children', allow_duplicate=True), Output('diabetes-pedigree-function-warning', 'children', allow_duplicate=True), # Added outputs for new warnings
     Output('glucose-warning', 'children', allow_duplicate=True), Output('blood-pressure-warning', 'children', allow_duplicate=True), Output('skin-thickness-warning', 'children', allow_duplicate=True),
     Output('insulin-warning', 'children', allow_duplicate=True), Output('bmi-warning', 'children', allow_duplicate=True), Output('age-warning', 'children', allow_duplicate=True),
     Output('prediction-output', 'children', allow_duplicate=True), Output('prediction-probability', 'children', allow_duplicate=True),
     Output('clinical-considerations-section', 'children', allow_duplicate=True), Output('input-vs-normal-graph', 'figure', allow_duplicate=True),
     Output('save-pdf-button', 'disabled', allow_duplicate=True),
     Output('save-pdf-button', 'title', allow_duplicate=True)],
    Input('reset-button', 'n_clicks'),
    [State('language-store', 'data'), State('theme-store', 'data')],
    prevent_initial_call=True
)
# Updated function signature for outputs
def reset_form_and_results(n_clicks, lang, theme):
    # Resets all input fields, warnings, and output/graph areas
    # Triggered by the 'Reset Form' button
    if n_clicks is None or n_clicks == 0: return dash.no_update # pragma: no cover

    # Get PDF button state/tooltip for after reset
    pdf_generally_possible = REPORTLAB_AVAILABLE
    pdf_button_title_after_reset = get_translation(lang, 'save_pdf_button_text')
    if not pdf_generally_possible: # pragma: no cover
        pdf_button_title_after_reset = get_translation(lang, 'alert_pdf_disabled')
    elif lang == 'ar' and not REPORTLAB_ARABIC_TOOLS_AVAILABLE: # pragma: no cover
        pdf_button_title_after_reset = get_translation(lang, 'alert_pdf_arabic_tools_missing')

    # Return a list of outputs to reset everything (Total 26 outputs now: 11 inputs + 11 warnings + 4 results)
    return ([
        None, None, None, # Reset input values (First Name, Last Name, DOB)
        None, None, # Reset input values (Pregnancies, DPF) - New
        None, None, None, # Reset input values (Glucose, BP, Skin Thickness)
        None, None, None, # Reset input values (Insulin, BMI, Age)
        "", "", "", # Clear warnings (First Name, Last Name, DOB)
        "", "", # Clear warnings (Pregnancies, DPF) - New
        "", "", "", # Clear warnings (Glucose, BP, Skin Thickness)
        "", "", "", # Clear warnings (Insulin, BMI, Age)
        "", # Clear prediction output
        "", # Clear prediction probability
        html.P(get_translation(lang, 'clinical_considerations_placeholder'), className="text-muted"), # Reset clinical considerations to initial placeholder
        create_default_graph(lang, theme), # Reset graph to default empty state based on theme/lang
        True, # Disable PDF button after reset (no prediction result to save)
        pdf_button_title_after_reset # Update PDF button tooltip
    ])


# load_history_table callback - MODIFIED to translate outcome text and handle N/A display explicitly
@app.callback(
    Output('history-table-content', 'children'),
    Input('current-page', 'data'),
    State('language-store', 'data') # Need language state to translate content
)
def load_history_table(current_page_name, lang):
    # Loads and displays the prediction history from the CSV file
    # Only runs when the history page is accessed
    if current_page_name != 'history': return None # Only update if on history page

    if lang not in TRANSLATIONS: lang = DEFAULT_LANG # pragma: no cover

    try:
        # Check if the predictions log file exists
        if os.path.exists(PREDICTIONS_CSV):
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(PREDICTIONS_CSV, encoding='utf-8')
            # Check if the DataFrame is not empty
            if not df.empty:
                # Reverse the DataFrame to show the latest predictions first
                df = df.iloc[::-1]
                # Map internal column names to translated header text keys
                # Added headers for Pregnancies and DPF
                headers_map = {
                    "Timestamp": "history_header_timestamp", "FirstName": "history_header_first_name",
                    "LastName": "history_header_last_name", "DateOfBirth": "history_header_dob",
                    "Pregnancies": "history_header_pregnancies", "DiabetesPedigreeFunction": "history_header_dpf", # Added new headers
                    "Glucose": "history_header_glucose", "BloodPressure": "history_header_bp",
                    "SkinThickness": "history_header_skinthickness", "Insulin": "history_header_insulin",
                    "BMI": "history_header_bmi", "Age": "history_header_age",
                    "PredictedOutcome": "history_header_outcome", # This header is translated below
                    "PredictedProbability": "history_header_probability" # This header is translated below
                }
                # Select only the columns that exist in the DataFrame and are in our headers map
                cols_to_display = [col for col in headers_map.keys() if col in df.columns]
                # Create a copy of the DataFrame with only the selected columns
                df_display = df[cols_to_display].copy()

                # --- Mapping for translating the PredictedOutcome cell content ---
                # Map the stored English key (which is the English display text) to the translation key
                outcome_translation_map = {
                    get_translation('en', 'pred_result_low_risk'): 'pred_result_low_risk',
                    get_translation('en', 'pred_result_high_risk'): 'pred_result_high_risk',
                    # Add fallback for older log entries if they used a slightly different string
                    "Low Diabetes Risk": 'pred_result_low_risk', # Older format
                    "Elevated Diabetes Risk": 'pred_result_high_risk', # Older format
                    "Low Risk (Model Prediction)": 'pred_result_low_risk', # Old format
                    "Elevated Risk (Model Prediction)": 'pred_result_high_risk', # Old format
                    # You might need to add other specific strings if they were ever logged
                }
                # --- End mapping ---

                # Format the Probability column to 3 decimal places if it exists
                if 'PredictedProbability' in cols_to_display:
                    # Convert to numeric first (handles potential non-numeric entries, setting them to NaN)
                    df_display['PredictedProbability'] = pd.to_numeric(df_display['PredictedProbability'], errors='coerce')
                    # Apply formatting, fill NaNs from coercion with 'N/A'
                    df_display['PredictedProbability'] = df_display['PredictedProbability'].map('{:.3f}'.format).fillna('N/A')

                 # Format DPF column to 2 decimal places if it exists
                if 'DiabetesPedigreeFunction' in cols_to_display: # pragma: no branch
                    df_display['DiabetesPedigreeFunction'] = pd.to_numeric(df_display['DiabetesPedigreeFunction'], errors='coerce')
                    df_display['DiabetesPedigreeFunction'] = df_display['DiabetesPedigreeFunction'].map('{:.2f}'.format).fillna('N/A')

                # Replace any remaining NaN values with 'N/A' for display
                # This is where N/A comes from for missing original inputs
                df_display = df_display.fillna('N/A')

                # Create the table header row with translated column names
                # Use get_translation with the headers_map
                table_header = [html.Thead(html.Tr([html.Th(get_translation(lang, headers_map.get(col, col))) for col in cols_to_display]))]


                # --- Create the table body rows with translated outcome text and explicit N/A handling ---
                table_body_rows = []
                for i in range(len(df_display)):
                    row_cells = []
                    for col in cols_to_display:
                        cell_value = df_display.iloc[i][col]
                        display_value = cell_value # Start with the value from the DataFrame

                        # --- Explicit check for None, pandas Na/NaN, or empty/whitespace string for display ---
                        # This adds robustness over just using df.fillna('N/A')
                        if pd.isna(cell_value) or cell_value is None or (isinstance(cell_value, str) and str(cell_value).strip() == ""):
                            display_value = 'N/A'
                        # --- End explicit check ---


                        # Check if this is the 'PredictedOutcome' column and the value is a string (and not 'N/A')
                        # Only attempt translation if the value is potentially a valid outcome key
                        if col == "PredictedOutcome" and isinstance(display_value, str) and display_value != 'N/A': # pragma: no branch
                            # Look up the stored English key in the translation map
                            # Use .strip() to handle potential leading/trailing whitespace from CSV
                            translation_key = outcome_translation_map.get(display_value.strip(), None)
                            if translation_key: # pragma: no branch
                                # If a matching translation key is found, get the translated text
                                translated_outcome_text = get_translation(lang, translation_key)
                                row_cells.append(html.Td(translated_outcome_text))
                            else: # pragma: no cover
                                # Fallback to the stored value if no translation key found (e.g., old format, error)
                                logger.warning(f"No translation map for history outcome key: '{display_value}' in language '{lang}'. Displaying raw value.")
                                row_cells.append(html.Td(display_value)) # Display the raw value if translation fails
                        else:
                            # For all other columns, display the value directly (already set to display_value)
                            row_cells.append(html.Td(display_value))
                    table_body_rows.append(html.Tr(row_cells)) # Add the row to the body

                table_body = [html.Tbody(table_body_rows)] # Wrap the rows in Tbody

                # Return the complete Dash Table component
                return dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True, responsive=True, className="mt-3")

        # If the file doesn't exist or is empty, return the "empty history" alert
        return dbc.Alert(get_translation(lang, 'history_table_empty'), color="info", className="mt-3")

    except pd.errors.EmptyDataError: # pragma: no cover
        # Specifically catch pandas EmptyDataError if the file is found but empty
        return dbc.Alert(get_translation(lang, 'history_table_empty'), color="info", className="mt-3")
    except Exception as e: # pragma: no cover
        # Catch any other unexpected errors during file reading or processing
        logger.error(f"Error loading history: {e}", exc_info=True)
        # Return a detailed error message for debugging
        return dbc.Alert(get_translation(lang, 'history_table_empty') + f" (Error: {type(e).__name__})", color="danger", className="mt-3")


# download_pdf callback (Corrected extraction of considerations)
@app.callback(
    Output("download-pdf", "data"),
    Output("pdf-error-output", "children"), # Output for temporary error alerts
    Input("save-pdf-button", "n_clicks"),
    [State('first-name', 'value'), State('last-name', 'value'), State('date-of-birth', 'value'),
     State('pregnancies', 'value'), State('diabetes-pedigree-function', 'value'), # Added States for new inputs
     State('glucose', 'value'), State('blood-pressure', 'value'), State('skin-thickness', 'value'),
     State('insulin', 'value'), State('bmi', 'value'), State('age', 'value'),
     State('prediction-output', 'children'), # Get the prediction result children (the Alert component)
     State('prediction-probability', 'children'), # Get the probability text children
     State('clinical-considerations-section', 'children'), # Get the clinical considerations section children (Div containing Ul/Ps)
     State('language-store', 'data')], # Need language state to get display text for PDF
    prevent_initial_call=True
)
# Updated function signature to include new States
def download_pdf(n_clicks_pdf, fname, lname, dob_str,
                 pregnancies, dpf, # Added new inputs
                 glucose, bp, st, insulin, bmi, age_input_val,
                 prediction_output_children, probability_output_children,
                 clinical_considerations_children, lang):
    # Triggered when the 'Save Report (PDF)' button is clicked
    if n_clicks_pdf is None or n_clicks_pdf == 0: # pragma: no cover
        return dash.no_update, dash.no_update # Don't do anything if not clicked

    if lang not in TRANSLATIONS: lang = DEFAULT_LANG # pragma: no cover

    # Check if PDF generation is even possible (ReportLab installed, Arabic tools if needed)
    alert_to_show = None
    if not REPORTLAB_AVAILABLE: # pragma: no cover
        alert_to_show = get_translation(lang, 'alert_pdf_disabled')
    elif lang == 'ar' and not REPORTLAB_ARABIC_TOOLS_AVAILABLE: # pragma: no cover
        alert_to_show = get_translation(lang, 'alert_pdf_arabic_tools_missing')
    if alert_to_show: # pragma: no cover
        # Return None for download data and an alert message
        return None, dbc.Alert(alert_to_show, color="warning", dismissable=True, duration=7000) # Alert disappears after 7s


    # Collect the patient input data to include in the PDF
    # Use the authoritative age determined during prediction if available, fallback to input field if needed
    authoritative_age_value = None
    if dob_str: # pragma: no branch
        calculated_age_from_dob = calculate_age(dob_str)
        if calculated_age_from_dob is not None: # pragma: no branch
            authoritative_age_value = calculated_age_from_dob
    if authoritative_age_value is None and age_input_val is not None and str(age_input_val).strip() != "": # pragma: no cover
        try:
            authoritative_age_value = int(float(str(age_input_val)))
        except (ValueError, TypeError): # pragma: no cover
            authoritative_age_value = None

    # Added Pregnancies and DPF to the inputs dictionary for PDF
    inputs_for_pdf_dict = {
        "FirstName": fname, "LastName": lname, "DateOfBirth": dob_str,
        "Pregnancies": pregnancies, "DiabetesPedigreeFunction": dpf, # Added new inputs
        "Glucose": glucose, "BloodPressure": bp, "SkinThickness": st,
        "Insulin": insulin, "BMI": bmi, "Age": authoritative_age_value
    }

    # Check if essential clinical data is present. A report without clinical inputs isn't useful.
    # This checks the 8 inputs that go into the model (including the new ones)
    essential_model_inputs_for_pdf_check = [
        inputs_for_pdf_dict["Pregnancies"], inputs_for_pdf_dict["DiabetesPedigreeFunction"], # Added new inputs
        inputs_for_pdf_dict["Glucose"], inputs_for_pdf_dict["BloodPressure"],
        inputs_for_pdf_dict["SkinThickness"], inputs_for_pdf_dict["Insulin"],
        inputs_for_pdf_dict["BMI"], inputs_for_pdf_dict["Age"]
    ]
    # Use any() to check if AT LEAST ONE of the essential inputs is NOT None/empty/whitespace
    has_any_essential_input = any(val is not None and (not isinstance(val, str) or str(val).strip() != "") for val in essential_model_inputs_for_pdf_check)

    if not has_any_essential_input: # pragma: no cover
         # If no essential clinical data is present, return a warning and prevent download
         return None, dbc.Alert(get_translation(lang, 'alert_pdf_no_clinical_data'), color="warning", dismissable=True, duration=5000)


    # Extract prediction text for PDF (needs the display text from the Alert)
    prediction_text_for_pdf = "N/A"
    # Check if prediction_output_children exists and is the Alert component
    if prediction_output_children and isinstance(prediction_output_children, dict) and prediction_output_children.get('type') == 'Alert': # pragma: no branch
        # Access the 'props' of the Alert, then its 'children' list
        alert_props_children = prediction_output_children.get('props', {}).get('children', [])
        # Assuming the Alert children are typically [Icon, Strong(Text), ...]
        if isinstance(alert_props_children, list) and len(alert_props_children) > 1: # pragma: no branch
            strong_element_candidate = alert_props_children[1] # Get the second child (expected Strong)
            if isinstance(strong_element_candidate, dict) and strong_element_candidate.get('type') == 'Strong': # pragma: no branch
                # Get the text content of the Strong element
                text_content = strong_element_candidate.get('props', {}).get('children', "N/A")
                # Handle case where children might be a list (e.g., ['Part1', 'Part2']) or a single string
                prediction_text_for_pdf = "".join(map(str, text_content)) if isinstance(text_content, list) else str(text_content)
                prediction_text_for_pdf = prediction_text_for_pdf.strip()
    elif isinstance(prediction_output_children, str) and prediction_output_children.strip(): # pragma: no cover
         # Fallback if prediction output is just a string (less common)
         prediction_text_for_pdf = prediction_output_children.strip()

    # Extract probability text for PDF (needs the display text from the Div)
    probability_text_for_pdf = str(probability_output_children) if probability_output_children is not None and isinstance(probability_output_children, str) else "N/A"
    probability_text_for_pdf = probability_text_for_pdf.strip()


    # Extract clinical considerations text strings from the hidden div
    considerations_list_for_pdf = []
    # Check if clinical_considerations_children is the main Div container
    if isinstance(clinical_considerations_children, dict) and clinical_considerations_children.get('type') == 'Div': # pragma: no branch
        div_children = clinical_considerations_children.get('props', {}).get('children', [])
        # Look for the hidden Div with id 'clinical-considerations-text-for-pdf'
        hidden_div_content = None
        for child in div_children: # pragma: no branch
             if isinstance(child, dict) and child.get('type') == 'Div' and child.get('props', {}).get('id') == 'clinical-considerations-text-for-pdf': # pragma: no branch
                  hidden_div_content = child.get('props', {}).get('children')
                  break

        # The children of the hidden div should be the list of strings generated in update_clinical_considerations
        if isinstance(hidden_div_content, list): # pragma: no branch
             considerations_list_for_pdf = hidden_div_content
        elif isinstance(hidden_div_content, str) and hidden_div_content.strip(): # pragma: no cover
             # Handle case where it might be a single string (e.g., the "none" message if Ul wasn't created)
             considerations_list_for_pdf = [hidden_div_content]
        else:
             # Fallback if hidden div content is unexpected (e.g., empty)
             logger.warning("Could not extract considerations text from hidden div.") # pragma: no cover


    # If the extracted list is empty after checking the hidden div, but the UI showed
    # the "no considerations" message (meaning the P element was returned by update_clinical_considerations),
    # then add that message to the list for the PDF.
    no_consid_text = get_translation(lang, 'clinical_considerations_none')
    is_ui_showing_no_consid_p = (isinstance(clinical_considerations_children, dict) and clinical_considerations_children.get('type') == 'P' and clinical_considerations_children.get('props', {}).get('children') == no_consid_text)

    if not considerations_list_for_pdf and is_ui_showing_no_consid_p: # pragma: no branch
         considerations_list_for_pdf = [no_consid_text] # Add the "no considerations" text

    # If no considerations were generated at all (inputs were missing or invalid and triggered a warning, not the considerations callback),
    # ensure the list is explicitly [no_consid_text] to avoid an empty section in the PDF.
    # This handles the case where predict failed before clinical_considerations updated.
    # We can check if prediction_text_for_pdf is "N/A" or an error message alert text.
    if not considerations_list_for_pdf and (prediction_text_for_pdf == "N/A" or "Error loading prediction components" in prediction_text_for_pdf or "Prediction model not available" in prediction_text_for_pdf): # pragma: no branch
         considerations_list_for_pdf = [no_consid_text]


    # Generate the PDF binary data using the collected information
      
    pdf_binary_data = generate_pdf_report(
        lang=lang,
        inputs=inputs_for_pdf_dict, # Pass dictionary including new inputs
        prediction_text=prediction_text_for_pdf,
        probability_text=probability_text_for_pdf,       # <-- Added a comma here
        considerations_list_of_strings=considerations_list_for_pdf # Pass the list of strings
    )
    # If PDF data was successfully generated
    if pdf_binary_data: # pragma: no cover
        # Create a filename based on timestamp
        pdf_filename = f"DiaRisk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        # Return the data dictionary for dcc.Download
        return dict(content=base64.b64encode(pdf_binary_data).decode(), filename=pdf_filename, type="application/pdf", base64=True), None # Return None for error output

    # If PDF generation failed (e.g., ReportLab error)
    # generate_pdf_report would have already logged the error.
    # Return None for download data and a generic error alert message
    return None, dbc.Alert(get_translation(lang, 'alert_pdf_generation_error'), color="danger", dismissable=True, duration=5000) # pragma: no cover


# --- Main execution block ---
if __name__ == '__main__': # pragma: no cover
    # Check if model loading failed and print fatal error message
    if model_load_error: # pragma: no cover
         logger.error(f"Application cannot start due to model/scaler loading issues: {model_load_error}")
         print(f"\n--- FATAL APPLICATION ERROR ---\n{model_load_error}\nPlease ensure '{MODEL_FILE}' and '{SCALER_FILE}' exist and are accessible, and 'catboost' is installed.\nApplication will not start.\n--------------------\n")
         # Exit is necessary if the model cannot be loaded, as the core functionality won't work.
         import sys
         sys.exit(1)

    # Log warnings if optional features (PDF/Arabic) are not fully available
    if not REPORTLAB_AVAILABLE: # pragma: no cover
        logger.warning("ReportLab library is not installed. PDF generation will be disabled.")
    elif REPORTLAB_AVAILABLE and not REPORTLAB_ARABIC_TOOLS_AVAILABLE: # pragma: no cover
        logger.warning("Full Arabic PDF support tools (arabic_reshaper, python-bidi, or font) are missing or failed to initialize. Arabic text in PDFs may not render correctly. Ensure Amiri-Regular.ttf is in /assets and arabic_reshaper, python-bidi are installed.")


    # Log CatBoost status if it wasn't the primary load error (it would be caught by model_load_error check anyway)
    if not CATBOOST_AVAILABLE and model_load_error is None: # pragma: no cover
         logger.warning("CatBoost library is not installed. Prediction button will be disabled.")


    # Start the Dash application server
    logger.info("Starting DiaRisk Dash application server...")
    # debug=True is useful during development; set to False for production
    # host='127.0.0.1' binds to localhost; use '0.0.0.0' to make it accessible externally (use caution)
    # port=8051 is the default Dash port
    app.run(debug=True, host='127.0.0.1', port=8051)