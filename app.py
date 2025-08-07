from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

app = Flask(__name__)

# Configuration
app.config['DATA_PATH'] = "predictive_maintenance_dataset.csv"
app.config['MAX_ROWS'] = 10000
COLOR_SCHEME = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'danger': '#e74c3c',
    'warning': '#f39c12',
    'background': '#f8f9fa'
}

# NEW: Added parameter ranges configuration
PARAMETER_RANGES = {
    'vibration': {'min': 0, 'max': 4, 'step': 0.1, 'unit': 'mm/s'},
    'temperature': {'min': 50, 'max': 75, 'step': 0.1, 'unit': '°C'},
    'pressure': {'min': 300, 'max': 400, 'step': 1, 'unit': 'kPa'},
    'rpm': {'min': 1500, 'max': 2000, 'step': 1, 'unit': ''}
}

def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(app.config['DATA_PATH'])
        df = df.dropna(how='all').loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
            except Exception as e:
                print(f"Timestamp error: {e}")
        
        if len(df) > app.config['MAX_ROWS']:
            df = df.sample(app.config['MAX_ROWS'])
        return df
    except Exception as e:
        raise Exception(f"Data loading failed: {str(e)}")

def calculate_health_scores(df):
    """Calculate equipment health scores (0-100 scale)"""
    try:
        health_features = [col for col in df.columns if any(x in col.lower() for x in ['vibration', 'temp', 'pressure', 'rpm'])]
        if not health_features:
            return None
        
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[health_features].fillna(0))
        weights = np.array([1.0/len(health_features)] * len(health_features))
        health_scores = 100 - (np.dot(scaled_values, weights) * 10)
        return np.clip(health_scores, 0, 100)
    except Exception as e:
        print(f"Health score error: {e}")
        return None

def detect_anomalies(df):
    """Identify anomalous behavior using Isolation Forest"""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if 'failure' in numeric_cols:
            numeric_cols.remove('failure')
        
        if not numeric_cols:
            return None
        
        clf = IsolationForest(contamination=0.05, random_state=42)
        anomalies = clf.fit_predict(df[numeric_cols].fillna(0))
        return [1 if x == -1 else 0 for x in anomalies]
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        return None

def create_visualizations(df, selected_columns):
    """Generate interactive visualizations"""
    visuals = []
    analysis_df = df[selected_columns].copy()
    numeric_cols = analysis_df.select_dtypes(include=['number']).columns.tolist()
    
    # Health Timeline
    health_scores = calculate_health_scores(df)
    if health_scores is not None and 'timestamp' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=health_scores,
            mode='lines',
            name='Health Score',
            line=dict(color=COLOR_SCHEME['primary'])
        ))
        
        if 'failure' in df.columns:
            failures = df[df['failure'] == 1]
            fig.add_trace(go.Scatter(
                x=failures['timestamp'],
                y=[100] * len(failures),
                mode='markers',
                name='Failure Events',
                marker=dict(color=COLOR_SCHEME['danger'], size=10)
            ))
        
        fig.update_layout(
            title='Equipment Health Timeline',
            xaxis_title='Date',
            yaxis_title='Health Score (0-100)',
            hovermode='x unified',
            template='plotly_white'
        )
        visuals.append(('health_timeline', fig.to_html()))
    
    # 3D Equipment State
    if len(numeric_cols) >= 3:
        fig = px.scatter_3d(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            z=numeric_cols[2],
            color='failure' if 'failure' in df.columns else None,
            title='3D Equipment State Analysis'
        )
        visuals.append(('3d_analysis', fig.to_html()))
    
    # Correlation Matrix
    if len(numeric_cols) > 1:
        fig = px.imshow(
            analysis_df[numeric_cols].corr(),
            text_auto=True,
            aspect="auto",
            title='Feature Correlation Matrix'
        )
        visuals.append(('correlation_matrix', fig.to_html()))
    
    return visuals

# NEW: Added visualization generation functions for test reports
def generate_health_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 40], 'color': COLOR_SCHEME['danger']},
                {'range': [40, 70], 'color': COLOR_SCHEME['warning']},
                {'range': [70, 100], 'color': COLOR_SCHEME['secondary']}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score}
        }
    ))
    return fig.to_html(full_html=False)

def generate_parameter_barchart(parameters):
    fig = go.Figure()
    for param in parameters:
        status_color = COLOR_SCHEME['danger'] if param['status'] == 'Critical' else COLOR_SCHEME['warning'] if param['status'] == 'Warning' else COLOR_SCHEME['secondary']
        fig.add_trace(go.Bar(
            x=[param['name']],
            y=[param['value']],
            name=param['name'],
            marker_color=status_color,
            text=[f"{param['value']}"],
            textposition='auto'
        ))
    fig.update_layout(
        title='Parameter Values',
        xaxis_title='Parameters',
        yaxis_title='Values',
        showlegend=False
    )
    return fig.to_html(full_html=False)

@app.route('/')
def dashboard():
    """Main dashboard view"""
    try:
        df = load_data()
        metrics = {
            'total_records': len(df),
            'date_range': {'start': df['timestamp'].min().strftime('%Y-%m-%d'), 
                          'end': df['timestamp'].max().strftime('%Y-%m-%d')} if 'timestamp' in df.columns else None,
            'failure_rate': round(df['failure'].mean() * 100, 2) if 'failure' in df.columns else None,
            'sensor_count': len(df.select_dtypes(include=['number']).columns)
        }
        health_scores = calculate_health_scores(df)
        health_chart = px.histogram(x=health_scores, nbins=20, title='Health Score Distribution').to_html() if health_scores is not None else None
        return render_template('dashboard.html', metrics=metrics, health_chart=health_chart, columns=df.columns.tolist())
    except Exception as e:
        return render_template('error.html', error_title="Dashboard Error", error_message=str(e))

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle analysis requests"""
    try:
        selected_columns = request.form.getlist('columns')
        if not selected_columns:
            raise ValueError("Please select at least one column")
        
        df = load_data()
        visuals = create_visualizations(df, selected_columns)
        
        return render_template('analysis.html',
                            visuals=visuals,
                            selected_columns=selected_columns)
    except Exception as e:
        return render_template('error.html',
                            error_title="Analysis Error",
                            error_message=str(e))

@app.route('/predict')
def predict():
    """Generate predictive insights"""
    try:
        df = load_data()
        health_scores = calculate_health_scores(df)
        anomalies = detect_anomalies(df)
        
        if 'timestamp' in df.columns and health_scores is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=health_scores,
                mode='lines',
                name='Health Score',
                line=dict(color=COLOR_SCHEME['primary'])
            ))
            
            if anomalies is not None:
                anomalies_df = df[df['anomaly'] == 1]
                fig.add_trace(go.Scatter(
                    x=anomalies_df['timestamp'],
                    y=[health_scores[i] for i in anomalies_df.index],
                    mode='markers',
                    name='Anomaly Detected',
                    marker=dict(color=COLOR_SCHEME['danger'], size=8)
                ))
            
            fig.update_layout(
                title='Predictive Maintenance Alerts',
                xaxis_title='Date',
                yaxis_title='Health Score',
                hovermode='x unified',
                template='plotly_white'
            )
            prediction_chart = fig.to_html()
        else:
            prediction_chart = None
        
        recommendations = []
        if health_scores is not None and np.mean(health_scores) < 60:
            recommendations.append({
                'severity': 'high',
                'message': "Low average health score detected. Schedule preventive maintenance."
            })
        
        return render_template('predictions.html',
                            prediction_chart=prediction_chart,
                            recommendations=recommendations)
    except Exception as e:
        return render_template('error.html',
                            error_title="Prediction Error",
                            error_message=str(e))

@app.route('/test-machine', methods=['GET', 'POST'])
def test_machine():
    """Machine testing functionality with enhanced reporting"""
    if request.method == 'POST':
        try:
            # Get and validate input values
            test_data = {}
            errors = []
            
            for param, config in PARAMETER_RANGES.items():
                try:
                    value = float(request.form.get(param, 0))
                    if not (config['min'] <= value <= config['max']):
                        errors.append(f"{param.capitalize()} must be between {config['min']}-{config['max']}{config['unit']}")
                    test_data[param] = value
                except ValueError:
                    errors.append(f"Invalid value for {param}")
            
            if errors:
                for error in errors:
                    flash(error)
                return redirect(url_for('test_machine'))
            
            # Process valid data
            df = load_data()
            results = {}
            
            for param, value in test_data.items():
                if param in df.columns:
                    mean = df[param].mean()
                    std = df[param].std()
                    z_score = (value - mean) / std if std != 0 else 0
                    status = 'Normal' if abs(z_score) < 2 else 'Warning' if abs(z_score) < 3 else 'Critical'
                    results[param] = {
                        'value': value,
                        'mean': mean,
                        'std': std,
                        'z_score': z_score,
                        'status': status,
                        'deviation': f"{((value - mean)/mean)*100:.1f}%" if mean != 0 else "N/A"
                    }
            
            # Calculate health score
            weights = {'vibration': 0.3, 'temperature': 0.25, 'pressure': 0.25, 'rpm': 0.2}
            health_score = sum(
                max(0, min(100, 100 - (abs(results[p]['z_score']) * 10))) * weights[p]
                for p in results
            )
            
            # NEW: Enhanced report data with visualizations
            parameters = [{
                'name': p,
                'value': results[p]['value'],
                'status': results[p]['status'],
                'deviation': results[p]['deviation'],
                'normal_range': f"{PARAMETER_RANGES[p]['min']}-{PARAMETER_RANGES[p]['max']}{PARAMETER_RANGES[p]['unit']}"
            } for p in results]
            
            report_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'parameters': parameters,
                'health_score': health_score,
                'recommendations': [
                    "Immediate maintenance required - high risk of failure" if health_score < 40 else
                    "Schedule preventive maintenance soon" if health_score < 70 else
                    "No immediate action needed",
                    *[f"{p.capitalize()} is abnormal ({results[p]['status']})" for p in results if results[p]['status'] != 'Normal']
                ],
                'visualizations': {
                    'health_gauge': generate_health_gauge(health_score),
                    'parameter_chart': generate_parameter_barchart(parameters)
                }
            }
            
            return render_template('test_results.html',
                                report_data=report_data,
                                color_scheme=COLOR_SCHEME)
            
        except Exception as e:
            flash("An error occurred during testing")
            return redirect(url_for('test_machine'))
    
    return render_template('test_machine.html',
                         parameter_ranges=PARAMETER_RANGES,
                         color_scheme=COLOR_SCHEME)

@app.route('/download-report')
def download_report():
    """Generate PDF report with visualizations"""
    try:
        report_data = json.loads(request.args.get('report_data', '{}'))
        buffer = io.BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        
        # Report header
        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, 750, "Equipment Health Test Report")
        p.setFont("Helvetica", 12)
        p.drawString(100, 730, f"Test Date: {report_data.get('timestamp', 'N/A')}")
        p.drawString(100, 710, f"Overall Health Score: {report_data.get('health_score', 0):.1f}/100")
        
        # Parameter results
        y = 670
        p.drawString(100, y, "Parameter Analysis:")
        y -= 20
        for param in report_data.get('parameters', []):
            p.drawString(120, y, f"{param['name'].capitalize()}: {param['value']} (Status: {param['status']})")
            p.drawString(350, y, f"Normal range: {param['normal_range']}")
            y -= 15
        
        # Recommendations
        y -= 20
        p.drawString(100, y, "Recommendations:")
        y -= 20
        for rec in report_data.get('recommendations', []):
            p.drawString(120, y, f"- {rec}")
            y -= 15
        
        p.showPage()
        p.save()
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="equipment_health_report.pdf")
    except Exception as e:
        flash(f"Error generating report: {str(e)}")
        return redirect(url_for('test_machine'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')