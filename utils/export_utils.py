"""
Export functionality for reports, routes, and visualizations
"""

import pandas as pd
import json
import base64
from io import BytesIO
import plotly.graph_objects as go
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def export_route_to_csv(route, delivery_locations, depots, filename='route.csv'):
    """Export route to CSV"""
    route_data = []
    
    # Add depot
    depot = depots.iloc[0]
    route_data.append({
        'stop_number': 0,
        'type': 'Depot',
        'delivery_id': 'DEPOT',
        'latitude': depot['latitude'],
        'longitude': depot['longitude'],
        'address': f"Depot at {depot['latitude']:.4f}, {depot['longitude']:.4f}"
    })
    
    # Add deliveries
    for i, idx in enumerate(route):
        loc = delivery_locations.iloc[idx]
        route_data.append({
            'stop_number': i + 1,
            'type': 'Delivery',
            'delivery_id': loc['delivery_id'],
            'latitude': loc['latitude'],
            'longitude': loc['longitude'],
            'address': loc.get('address', f"Location {loc['delivery_id']}")
        })
    
    # Add return to depot
    route_data.append({
        'stop_number': len(route) + 1,
        'type': 'Depot',
        'delivery_id': 'DEPOT',
        'latitude': depot['latitude'],
        'longitude': depot['longitude'],
        'address': f"Return to Depot"
    })
    
    df = pd.DataFrame(route_data)
    csv = df.to_csv(index=False)
    return csv


def export_metrics_to_excel(metrics_dict, filename='metrics.xlsx'):
    """Export metrics to Excel"""
    # Create Excel writer
    buffer = BytesIO()
    
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            'Metric': list(metrics_dict.keys()),
            'Value': [str(v) for v in metrics_dict.values()]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    buffer.seek(0)
    return buffer.getvalue()


def create_pdf_report(title, sections, filename='report.pdf'):
    """Create PDF report"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = styles['Title']
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Sections
    for section_title, section_content in sections:
        heading = styles['Heading2']
        story.append(Paragraph(section_title, heading))
        story.append(Spacer(1, 0.2*inch))
        
        if isinstance(section_content, pd.DataFrame):
            # Convert DataFrame to table
            data = [section_content.columns.tolist()] + section_content.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#4472C4'),
                ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), '#E7E6E6'),
                ('GRID', (0, 0), (-1, -1), 1, 'black')
            ]))
            story.append(table)
        else:
            story.append(Paragraph(str(section_content), styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def export_plot_as_image(fig, format='png'):
    """Export plotly figure as image"""
    try:
        import kaleido
        img_bytes = fig.to_image(format=format, width=1200, height=800)
        return img_bytes
    except:
        # Fallback: return base64 encoded HTML
        html = fig.to_html(include_plotlyjs='cdn')
        return html.encode()


def create_shareable_link(scenario_data):
    """Create shareable link data (base64 encoded)"""
    data_json = json.dumps(scenario_data)
    encoded = base64.b64encode(data_json.encode()).decode()
    return encoded


def load_from_shareable_link(encoded_data):
    """Load scenario from shareable link"""
    decoded = base64.b64decode(encoded_data.encode()).decode()
    return json.loads(decoded)

