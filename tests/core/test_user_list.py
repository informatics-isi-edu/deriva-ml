"""Regression test for ``DerivaML.user_list()``.

``user_list()`` was once deleted as "dead code" (zero callers *inside*
deriva-ml) even though external subclasses (e.g. EyeAI) depend on it. This
test pins the public method's existence and contract so the deletion cannot
silently recur.

The user table (``public.ERMrest_Client``) lives outside the domain/ML schema
search path, so ``get_table_as_dict("ERMrest_Client")`` cannot reach it —
``user_list()`` is the supported accessor. See
``core/mixins/path_builder.py``.
"""


class TestUserList:
    def test_user_list_contract(self, test_ml):
        """user_list() returns a list of {'ID', 'Full_Name'} dicts.

        Asserts the shape/contract rather than specific user content, since
        the rows in public.ERMrest_Client vary by environment. ERMrest
        auto-populates this table with at least the catalog's creator, so the
        list is expected to be non-empty against a real catalog.
        """
        users = test_ml.user_list()

        assert isinstance(users, list)
        # ERMrest_Client always holds at least the creating client.
        assert len(users) >= 1
        for user in users:
            assert set(user.keys()) == {"ID", "Full_Name"}

    def test_user_list_method_exists(self, test_ml):
        """Guard against re-deletion: the public method must be present."""
        assert hasattr(test_ml, "user_list")
        assert callable(test_ml.user_list)
