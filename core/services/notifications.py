"""
通知システム（Slack/Email inspired）

Implements: F-NOTIFY-001
設計思想:
- 複数チャネル対応
- テンプレート
- 通知履歴
"""

from __future__ import annotations

import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    """通知"""
    id: str
    channel: str
    subject: str
    message: str
    sent_at: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None


class NotificationService:
    """
    通知サービス
    
    Features:
    - Slack/Email/Webhook対応
    - テンプレート
    - 通知履歴
    
    Example:
        >>> notifier = NotificationService()
        >>> notifier.send_slack("Training complete!", webhook_url="...")
    """
    
    def __init__(self):
        self._history: List[Notification] = []
        self._counter = 0
    
    def send_slack(
        self,
        message: str,
        webhook_url: str,
        channel: Optional[str] = None,
    ) -> Notification:
        """Slack通知"""
        self._counter += 1
        notif_id = f"notif_{self._counter:05d}"
        
        payload = {"text": message}
        if channel:
            payload["channel"] = channel
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                success = response.status == 200
            
            notif = Notification(
                id=notif_id,
                channel="slack",
                subject="Slack Message",
                message=message,
                success=success,
            )
            
        except Exception as e:
            notif = Notification(
                id=notif_id,
                channel="slack",
                subject="Slack Message",
                message=message,
                success=False,
                error=str(e),
            )
        
        self._history.append(notif)
        return notif
    
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        smtp_host: str = "localhost",
        smtp_port: int = 25,
        from_addr: str = "chemml@localhost",
    ) -> Notification:
        """Email通知"""
        self._counter += 1
        notif_id = f"notif_{self._counter:05d}"
        
        try:
            msg = MIMEMultipart()
            msg["From"] = from_addr
            msg["To"] = to
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.send_message(msg)
            
            notif = Notification(
                id=notif_id,
                channel="email",
                subject=subject,
                message=body,
                success=True,
            )
            
        except Exception as e:
            notif = Notification(
                id=notif_id,
                channel="email",
                subject=subject,
                message=body,
                success=False,
                error=str(e),
            )
        
        self._history.append(notif)
        return notif
    
    def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any],
    ) -> Notification:
        """Webhook通知"""
        self._counter += 1
        notif_id = f"notif_{self._counter:05d}"
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                success = 200 <= response.status < 300
            
            notif = Notification(
                id=notif_id,
                channel="webhook",
                subject="Webhook",
                message=json.dumps(payload),
                success=success,
            )
            
        except Exception as e:
            notif = Notification(
                id=notif_id,
                channel="webhook",
                subject="Webhook",
                message=json.dumps(payload),
                success=False,
                error=str(e),
            )
        
        self._history.append(notif)
        return notif
    
    def notify_experiment_complete(
        self,
        experiment_name: str,
        metrics: Dict[str, float],
        webhook_url: Optional[str] = None,
    ) -> None:
        """実験完了通知"""
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        message = f"🎉 Experiment '{experiment_name}' completed!\n📊 Metrics: {metrics_str}"
        
        if webhook_url:
            self.send_slack(message, webhook_url)
        
        logger.info(message)
    
    def get_history(self, channel: Optional[str] = None) -> List[Notification]:
        """通知履歴"""
        if channel:
            return [n for n in self._history if n.channel == channel]
        return self._history
