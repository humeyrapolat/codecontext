import unittest

from app.state import AppState


class AppStateTests(unittest.TestCase):
    def test_set_repository_marks_it_active(self):
        state = AppState()

        state.set_repository("repo-1", "https://github.com/acme/repo.git", object())

        self.assertTrue(state.agent_ready)
        self.assertEqual(state.active_repo_id, "repo-1")
        self.assertEqual(state.require_repository().repo_id, "repo-1")

    def test_can_store_multiple_repositories(self):
        state = AppState()

        first_agent = object()
        second_agent = object()
        state.set_repository("repo-1", "https://github.com/acme/one.git", first_agent)
        state.set_repository("repo-2", "https://github.com/acme/two.git", second_agent)

        self.assertEqual(len(state.list_repositories()), 2)
        self.assertIs(state.require_agent("repo-1"), first_agent)
        self.assertIs(state.require_agent("repo-2"), second_agent)
        self.assertEqual(state.active_repo_id, "repo-2")

    def test_require_repository_fails_when_missing(self):
        state = AppState()

        with self.assertRaises(RuntimeError):
            state.require_repository("missing")


if __name__ == "__main__":
    unittest.main()
