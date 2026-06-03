from __future__ import annotations

import unittest

from evopolicygym.envs.gym import space


class Unicode:
    pass


class Float:
    pass


class Integer:
    pass


class AnyDict:
    pass


class AnyBox:
    shape = (-1, -1, 3)
    dtype = "uint8"
    low = 0
    high = 255


class GymSpaceTest(unittest.TestCase):
    def test_browsergym_unicode_action_has_noop_sample(self) -> None:
        action = space.sample(Unicode())

        self.assertEqual(action, "noop()")
        self.assertEqual(space.action(Unicode(), action), ("noop()", False))
        self.assertEqual(space.action(Unicode(), 1), ("", True))

    def test_browsergym_scalar_and_mapping_spaces_have_safe_defaults(self) -> None:
        self.assertEqual(space.sample(Float()), 0.0)
        self.assertEqual(space.sample(Integer()), 0)
        self.assertEqual(space.sample(AnyDict()), {})
        self.assertEqual(space.action(Float(), "1.5"), (1.5, False))
        self.assertEqual(space.action(Integer(), "2"), (2, False))
        self.assertEqual(space.action(AnyDict(), {"x": 1}), ({"x": 1}, False))

    def test_browsergym_anybox_image_schema_uses_external_storage(self) -> None:
        schema = space.schema(AnyBox())

        self.assertEqual(schema["type"], "Image")
        self.assertEqual(schema["shape"], [-1, -1, 3])
        self.assertEqual(schema["storage"], "external")


if __name__ == "__main__":
    unittest.main()
