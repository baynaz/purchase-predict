from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

class TestKedroRun:
    def test_kedro_session_starts(self):
        bootstrap_project(Path.cwd())
        with KedroSession.create(project_path=Path.cwd()) as session:
            assert session is not None
