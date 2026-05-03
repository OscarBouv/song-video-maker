import json
import logging
from pathlib import Path
from typing import Optional

from instagrapi import Client
from instagrapi.exceptions import LoginRequired

from config import INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD, INSTAGRAM_SESSION_FILE

logger = logging.getLogger(__name__)

class InstagramPublisher:
    """Handles publishing Reels to Instagram using instagrapi."""

    def __init__(self):
        self.cl = Client()
        self.username = INSTAGRAM_USERNAME
        self.password = INSTAGRAM_PASSWORD
        self.session_file = INSTAGRAM_SESSION_FILE

    def login(self) -> bool:
        """Authenticate with Instagram, using cached session if available."""
        if not self.username or not self.password:
            logger.error("Instagram credentials not found in environment.")
            return False

        if self.session_file.exists():
            try:
                self.cl.load_settings(self.session_file)
                self.cl.login(self.username, self.password)
                try:
                    self.cl.get_timeline_feed()  # Check if session is still valid
                except LoginRequired:
                    logger.info("Session expired, logging in again...")
                    self.cl.login(self.username, self.password)
            except Exception as e:
                logger.warning(f"Failed to load session: {e}. Logging in from scratch.")
                self.cl.login(self.username, self.password)
        else:
            self.cl.login(self.username, self.password)

        self.cl.dump_settings(self.session_file)
        return True

    def publish_reel(self, video_path: Path, caption: str) -> Optional[str]:
        """Upload and publish a video as a Reel.

        Returns the media ID on success, None on failure.
        """
        if not self.login():
            return None

        try:
            logger.info(f"Uploading Reel: {video_path}")
            media = self.cl.clip_upload(
                path=video_path,
                caption=caption
            )
            return media.id
        except Exception as e:
            logger.error(f"Failed to publish Reel: {e}")
            return None

def publish_to_instagram(video_path: Path, caption: str) -> bool:
    """Convenience function to publish a Reel."""
    publisher = InstagramPublisher()
    media_id = publisher.publish_reel(video_path, caption)
    if media_id:
        print(f"✓ Successfully published to Instagram! Media ID: {media_id}")
        return True
    else:
        print("✗ Failed to publish to Instagram.")
        return False
