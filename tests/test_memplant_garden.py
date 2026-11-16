"""Living Memory Garden source-security and simulation invariants."""
import json
import unittest
from experiments.memplant.memorygarden.garden import (
    World,Invention,fixtures,evaluate,make_prompt)


class GardenTests(unittest.TestCase):
    def test_oracle_mechanisms_function_for_all_four_worlds(self):
        for world in fixtures():
            expected=Invention(
                world.name,"rigid" if world.heat_hardens else "compliant",
                "ceiling" if world.dust_floats else "floor",
                "heat_then_shake" if world.heat_hardens else "shake_then_heat",
                "mechanistically appropriate")
            self.assertTrue(evaluate(world,expected)["passed"])
            wrong=Invention(expected.name,expected.frame,
                            "floor" if expected.collector=="ceiling" else "ceiling",
                            expected.sequence,expected.explanation)
            self.assertFalse(evaluate(world,wrong)["passed"])

    def test_no_memory_prompt_identical_for_opposite_physics(self):
        for world in fixtures():
            self.assertEqual(make_prompt(world,"none"),
                             make_prompt(world.inverted(),"none"))
            self.assertNotEqual(make_prompt(world,"text_card"),
                                make_prompt(world,"wrong_card"))
        with self.assertRaises(ValueError):
            make_prompt(fixtures()[0],"full_text")

    def test_json_is_safely_parsed_and_invalid_values_are_rejected(self):
        row=json.dumps({"name":"Crystalline trap","frame":"rigid",
                        "collector":"ceiling","sequence":"heat_then_shake",
                        "explanation":"A simple recorder."})
        self.assertEqual(Invention.from_text(row).collector,"ceiling")
        with self.assertRaises(ValueError):
            Invention.from_text("No valid object")
        with self.assertRaises(ValueError):
            Invention.from_text(row.replace('"ceiling"','"rocket"'))

if __name__=="__main__":
    unittest.main()
