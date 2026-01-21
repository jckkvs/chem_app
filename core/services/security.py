"""
セキュリティユーティリティ

Implements: F-SECURITY-001
設計思想:
- 入力サニタイズ
- 認証/認可
- 監査ログ
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditLog:
    """監査ログ"""
    timestamp: str
    user: str
    action: str
    resource: str
    status: str
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """
    セキュリティマネージャー
    
    Features:
    - 入力検証
    - トークン生成
    - 監査ログ
    
    Example:
        >>> security = SecurityManager()
        >>> token = security.generate_token()
        >>> security.validate_smiles(user_input)
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.audit_logs: List[AuditLog] = []
    
    def generate_token(self, length: int = 32) -> str:
        """セキュアトークン生成"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> str:
        """パスワードハッシュ"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_value = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt.encode(),
            100000,
        ).hex()
        
        return f"{salt}${hash_value}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """パスワード検証"""
        try:
            salt, hash_value = hashed.split('$')
            expected = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt.encode(),
                100000,
            ).hex()
            return hmac.compare_digest(hash_value, expected)
        except Exception:
            return False
    
    def sanitize_smiles(self, smiles: str) -> str:
        """SMILESをサニタイズ"""
        # 許可された文字のみ
        allowed = set('CNOSPFClBrI[]()=#-+@/\\0123456789cnops')
        sanitized = ''.join(c for c in smiles if c in allowed)
        return sanitized[:500]  # 長さ制限
    
    def validate_input(self, value: str, max_length: int = 1000) -> bool:
        """入力検証"""
        if not value or len(value) > max_length:
            return False
        
        # 危険なパターンをチェック
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'on\w+=',
            r'\${',
            r'{{',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    def log_audit(
        self,
        user: str,
        action: str,
        resource: str,
        status: str = 'success',
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """監査ログ記録"""
        log_entry = AuditLog(
            timestamp=datetime.now().isoformat(),
            user=user,
            action=action,
            resource=resource,
            status=status,
            details=details or {},
        )
        
        self.audit_logs.append(log_entry)
        
        if status == 'failure':
            logger.warning(f"Security: {action} on {resource} by {user} - FAILED")
    
    def get_audit_logs(
        self,
        user: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """監査ログ取得"""
        logs = self.audit_logs
        
        if user:
            logs = [l for l in logs if l.user == user]
        if action:
            logs = [l for l in logs if l.action == action]
        
        return logs[-limit:]
    
    def create_signature(self, data: str) -> str:
        """データ署名"""
        signature = hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()
        return signature
    
    def verify_signature(self, data: str, signature: str) -> bool:
        """署名検証"""
        expected = self.create_signature(data)
        return hmac.compare_digest(expected, signature)
