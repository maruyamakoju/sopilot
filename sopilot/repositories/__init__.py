"""Repository layer -- focused persistence modules.

Each repository encapsulates a cohesive set of database operations and
depends only on a ``connect`` callable (context-manager factory), not on
the full ``Database`` class.
"""

from sopilot.repositories.admin_repository import AdminRepository
from sopilot.repositories.base import ConnectFactory, RepositoryBase
from sopilot.repositories.score_repository import ScoreRepository
from sopilot.repositories.task_profile_repository import TaskProfileRepository
from sopilot.repositories.video_repository import VideoRepository

__all__ = [
    "AdminRepository",
    "ConnectFactory",
    "RepositoryBase",
    "ScoreRepository",
    "TaskProfileRepository",
    "VideoRepository",
]
