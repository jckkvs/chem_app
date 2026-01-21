"""
ジョブスケジューラー（Celery/APScheduler inspired）

Implements: F-SCHED-001
設計思想:
- タスクキュー
- 定期実行
- 優先度管理
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """ジョブ"""
    id: str
    func: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __lt__(self, other):
        return self.priority > other.priority  # 高い優先度が先


class JobScheduler:
    """
    ジョブスケジューラー（Celery inspired）
    
    Features:
    - 優先度付きキュー
    - バックグラウンド実行
    - ステータス追跡
    
    Example:
        >>> scheduler = JobScheduler()
        >>> scheduler.start()
        >>> job_id = scheduler.submit(my_func, args=(1, 2))
        >>> scheduler.get_status(job_id)
    """
    
    def __init__(self, n_workers: int = 2):
        self.n_workers = n_workers
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._jobs: Dict[str, Job] = {}
        self._workers: List[threading.Thread] = []
        self._running = False
        self._job_counter = 0
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """ワーカー開始"""
        self._running = True
        for i in range(self.n_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)
        logger.info(f"Started {self.n_workers} workers")
    
    def stop(self) -> None:
        """ワーカー停止"""
        self._running = False
        for worker in self._workers:
            worker.join(timeout=1)
        logger.info("Scheduler stopped")
    
    def _worker_loop(self) -> None:
        """ワーカーループ"""
        while self._running:
            try:
                job = self._queue.get(timeout=1)
                self._execute_job(job)
            except queue.Empty:
                continue
    
    def _execute_job(self, job: Job) -> None:
        """ジョブ実行"""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        
        try:
            job.result = job.func(*job.args, **job.kwargs)
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.error = str(e)
            job.status = JobStatus.FAILED
            logger.error(f"Job {job.id} failed: {e}")
        finally:
            job.completed_at = datetime.now()
    
    def submit(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> str:
        """ジョブ投入"""
        with self._lock:
            self._job_counter += 1
            job_id = f"job_{self._job_counter:05d}"
        
        job = Job(
            id=job_id,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
        )
        
        self._jobs[job_id] = job
        self._queue.put(job)
        
        logger.info(f"Submitted job {job_id}")
        return job_id
    
    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """ステータス取得"""
        job = self._jobs.get(job_id)
        return job.status if job else None
    
    def get_result(self, job_id: str, timeout: float = None) -> Any:
        """結果取得（ブロッキング）"""
        start = time.time()
        while True:
            job = self._jobs.get(job_id)
            if job and job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                if job.status == JobStatus.FAILED:
                    raise RuntimeError(job.error)
                return job.result
            
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Job {job_id} timed out")
            
            time.sleep(0.1)
    
    def cancel(self, job_id: str) -> bool:
        """ジョブキャンセル"""
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True
        return False
    
    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """ジョブ一覧"""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs
