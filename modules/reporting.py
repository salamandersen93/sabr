# reporting_refactored.py
import os
from datetime import datetime
from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

styles = getSampleStyleSheet()
styleN = styles['Normal']
bullet_style = styles['Normal']
bullet_style.leftIndent = 10

def dict_to_bullets(param_dict, n_cols=3):
    items = [Paragraph(f"• <b>{k}</b>: {v}", bullet_style) for k, v in param_dict.items()]
    # pad to multiple of n_cols
    while len(items) % n_cols != 0:
        items.append("")
    # reshape
    rows = [items[i:i+n_cols] for i in range(0, len(items), n_cols)]
    t = Table(rows, colWidths=[180]*n_cols)
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    return t

class BioreactorPDFReport:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        # Use unique style names to avoid KeyError
        self.styles.add(
            ParagraphStyle(
                name='CustomHeading1',
                fontSize=14,
                leading=16,
                spaceAfter=12,
                spaceBefore=12,
                alignment=1
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='CustomHeading2',
                fontSize=12,
                leading=14,
                spaceAfter=8,
                spaceBefore=8,
                textColor=colors.darkblue
            )
        )
        self.styles.add(
            ParagraphStyle(
                name='CustomBodyText',
                fontSize=10,
                leading=12,
                spaceAfter=4
            )
        )
    
    def _df_to_table(self, df: pd.DataFrame):
        """Convert a DataFrame to a ReportLab Table object."""
        data = [list(df.columns)] + df.values.tolist()
        table = Table(data, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return table

    def _fig_to_image(self, fig):
        """Convert a matplotlib figure to a ReportLab Image object."""
        buf = BytesIO()
        fig.savefig(buf, format='PNG', bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=400, height=250)  # adjust size as needed
        return img

    def generate_summary_pdf(self,results: dict,telemetry_df: pd.DataFrame,
                             ai_summary: str,faults: dict,param_config: dict,
                             figures: list = None,filename: str = None) -> str:
        """Generate a formal bioreactor simulation report."""
        run_id = results.get('run_id', 'unknown')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = filename or f"bioreactor_report_{run_id}.pdf"
        pdf_path = os.path.join(filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []

        # generate title
        story.append(Paragraph(f"Bioreactor Simulation Report: {run_id}", self.styles['Heading1']))
        story.append(Spacer(1, 12))

        # run information and metadata
        story.append(Paragraph("Run Information", self.styles['Heading2']))
        run_info_table = self._df_to_table(pd.DataFrame([{
            "Timestamp": timestamp,
            "Final Titer [mg/mL]": results.get("final_titer", "N/A"),
            "Final Biomass [g/L]": results.get("final_biomass", "N/A")}]))
        story.append(run_info_table)
        story.append(Spacer(1, 12))

        # display input parameters (kinetics and default values)
        story.append(Paragraph("Input Parameters", self.styles['Heading2']))
        for key, params in param_config.items():
            if key != 'FAULT_TEMPLATES':
                if key == 'KINETIC_PARAMS':
                    story.append(Paragraph("Kinetics:", self.styles['Heading3']))
                    story.append(dict_to_bullets(params))
                else:
                    story.append(Paragraph(f"{key}:", self.styles['BodyText']))
                    df_params = pd.DataFrame([params]) if isinstance(params, dict) else pd.DataFrame(params)
                    story.append(self._df_to_table(df_params))
                story.append(Spacer(1, 6))

        # display fault injection data
        story.append(Paragraph("Faults Injected", self.styles['Heading2']))
        if not faults:
            story.append(Paragraph("No faults were injected.", self.styles['BodyText']))
        else:
            if isinstance(faults, dict):
                if faults.get("type") == "standard":
                    story.append(Paragraph("No faults were injected (standard run).", self.styles['BodyText']))
                else:
                    df_faults = pd.DataFrame([faults])
                    story.append(self._df_to_table(df_faults))
            elif isinstance(faults, list):
                df_faults = pd.DataFrame(faults)
                story.append(self._df_to_table(df_faults))
            else:
                story.append(Paragraph(f"(Unexpected faults format: {type(faults)})", self.styles['BodyText']))
            story.append(Spacer(1, 12))

        # visualize AI summary
        # TODO: have another AI agent rate the first agent's summary with
        # respect to how well it identifies the injected fault
        story.append(Paragraph("AI Summary / Troubleshooting", self.styles['Heading2']))
        story.append(Paragraph(ai_summary, self.styles['BodyText']))
        story.append(Spacer(1, 12))

        # take a sample of the telemetry data to show in the report
        story.append(Paragraph("Telemetry Sample (first 10 rows)", self.styles['Heading2']))
        story.append(self._df_to_table(telemetry_df.head(10)))
        story.append(Spacer(1, 12))

        # display telemetry and anomaly detection figures
        if figures:
            story.append(Paragraph("Telemetry & Anomaly Plots", self.styles['Heading2']))
            for fig in figures:
                story.append(self._fig_to_image(fig))
                story.append(Spacer(1, 12))

        doc.build(story)
        return pdf_path