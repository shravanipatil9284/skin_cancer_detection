from fpdf import FPDF
from datetime import datetime
import os


def generate_report(user_name, result, confidence, image_path):
    """
    Generates a downloadable PDF report for a prediction
    """

    reports_dir = "static/reports"
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = os.path.join(reports_dir, filename)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Skin Cancer Detection Report", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Patient Name: {user_name}", ln=True)
    pdf.cell(0, 8, f"Prediction Result: {result}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence}%", ln=True)
    pdf.cell(0, 8, f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.multi_cell(
        0, 7,
        "Disclaimer: This result is generated using a deep learning model and "
        "is not a medical diagnosis. Please consult a dermatologist for clinical evaluation."
    )

    pdf.output(filepath)

    return filename
