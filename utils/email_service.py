#!/usr/bin/env python3
"""
Email service for sending intelligence reports
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending intelligence reports via email"""
    
    def __init__(self, smtp_server: str = None, smtp_port: int = None, 
                 username: str = None, password: str = None):
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.username = username or os.getenv('SMTP_USERNAME')
        self.password = password or os.getenv('SMTP_PASSWORD')
        
        if not all([self.username, self.password]):
            logger.warning("Email credentials not configured. Email functionality will be disabled.")
    
    def send_intelligence_report(self, 
                               recipient_email: str,
                               analysis_id: str,
                               intelligence_summary: dict,
                               attachments: List[str] = None) -> bool:
        """Send intelligence report via email"""
        
        if not all([self.username, self.password]):
            logger.error("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient_email
            msg['Subject'] = f"Ghost Hunter Intelligence Report - Analysis {analysis_id[:8]}"
            
            # Create email body
            body = self._create_email_body(analysis_id, intelligence_summary)
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if Path(file_path).exists():
                        self._add_attachment(msg, file_path)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.username, recipient_email, text)
            server.quit()
            
            logger.info(f"Intelligence report sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _create_email_body(self, analysis_id: str, intelligence_summary: dict) -> str:
        """Create HTML email body for intelligence report"""
        
        threat_level = intelligence_summary.get('threat_assessment', 'UNKNOWN')
        executive_summary = intelligence_summary.get('executive_summary', 'No summary available')
        key_findings = intelligence_summary.get('key_findings', [])
        recommendations = intelligence_summary.get('recommendations', [])
        
        # Determine threat color
        threat_color = '#ef4444' if 'HIGH' in threat_level else '#f59e0b' if 'MODERATE' in threat_level else '#10b981'
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Ghost Hunter Intelligence Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f8fafc; }}
                .container {{ max-width: 800px; margin: 0 auto; background-color: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
                .header {{ background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%); color: white; padding: 30px; border-radius: 8px 8px 0 0; }}
                .header h1 {{ margin: 0; font-size: 28px; font-weight: bold; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                .content {{ padding: 30px; }}
                .threat-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; color: white; background-color: {threat_color}; }}
                .section {{ margin-bottom: 30px; }}
                .section h2 {{ color: #1e293b; font-size: 20px; margin-bottom: 15px; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
                .summary {{ background-color: #f1f5f9; padding: 20px; border-radius: 6px; border-left: 4px solid #0891b2; }}
                .findings {{ list-style: none; padding: 0; }}
                .findings li {{ background-color: #fef7f0; padding: 12px; margin-bottom: 8px; border-radius: 4px; border-left: 3px solid #f59e0b; }}
                .recommendations {{ list-style: none; padding: 0; }}
                .recommendations li {{ background-color: #f0fdf4; padding: 12px; margin-bottom: 8px; border-radius: 4px; border-left: 3px solid #10b981; }}
                .footer {{ background-color: #f8fafc; padding: 20px; border-radius: 0 0 8px 8px; text-align: center; color: #64748b; font-size: 12px; }}
                .metadata {{ background-color: #f8fafc; padding: 15px; border-radius: 6px; margin-bottom: 20px; }}
                .metadata strong {{ color: #1e293b; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🛰️ GHOST HUNTER</h1>
                    <p>Maritime Intelligence Report</p>
                </div>
                
                <div class="content">
                    <div class="metadata">
                        <strong>Analysis ID:</strong> {analysis_id}<br>
                        <strong>Generated:</strong> {intelligence_summary.get('timestamp', 'Unknown')}<br>
                        <strong>Threat Level:</strong> <span class="threat-badge">{threat_level}</span>
                    </div>
                    
                    <div class="section">
                        <h2>📋 Executive Summary</h2>
                        <div class="summary">
                            {executive_summary}
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>🔍 Key Findings</h2>
                        <ul class="findings">
        """
        
        for finding in key_findings:
            html_body += f"<li>{finding}</li>"
        
        html_body += """
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h2>💡 Recommendations</h2>
                        <ul class="recommendations">
        """
        
        for recommendation in recommendations:
            html_body += f"<li>{recommendation}</li>"
        
        html_body += f"""
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This report was generated by the Ghost Hunter Maritime Surveillance System.</p>
                    <p>For questions or support, please contact your system administrator.</p>
                    <p><strong>CONFIDENTIAL:</strong> This report contains sensitive intelligence information.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_body
    
    def _add_attachment(self, msg: MIMEMultipart, file_path: str):
        """Add file attachment to email"""
        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            
            filename = Path(file_path).name
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            logger.error(f"Failed to add attachment {file_path}: {e}")
    
    def test_connection(self) -> bool:
        """Test SMTP connection"""
        if not all([self.username, self.password]):
            return False
        
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.quit()
            return True
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False