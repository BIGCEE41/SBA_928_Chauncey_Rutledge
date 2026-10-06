import csv
import json
import tempfile
import unittest
from pathlib import Path

from sba_928_chauncey_rutledge.training import (
    PROMPT_FILE,
    _target,
    build_examples,
    load_tickets,
    split_tickets,
)


class TrainingDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "tickets.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "Ticket Subject",
                    "Ticket Description",
                    "Product Purchased",
                    "Ticket Type",
                    "Ticket Priority",
                    "Ticket Channel",
                    "Customer Satisfaction Rating",
                    "Customer Name",
                    "Customer Email",
                    "Customer Age",
                    "Gender",
                ],
            )
            writer.writeheader()
            for index in range(10):
                writer.writerow(
                    {
                        "Ticket Subject": f"Return request {index}",
                        "Ticket Description": (
                            "I need help with {product_purchased}.\n"
                            "If you need to change an existing product.\n"
                            "I need help with {product_purchased}.\n"
                            f"Please contact customer{index}@example.com or 555-123-4567"
                        ),
                        "Product Purchased": "Widget",
                        "Ticket Type": "Refund",
                        "Ticket Priority": "Medium",
                        "Ticket Channel": "Email",
                        "Customer Satisfaction Rating": str(index % 5 + 1),
                        "Customer Name": f"Customer {index}",
                        "Customer Email": f"customer{index}@example.com",
                        "Customer Age": str(20 + index),
                        "Gender": "unknown",
                    }
                )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_tickets_redacts_direct_contact_and_omits_identity_fields(self) -> None:
        tickets = load_tickets(self.csv_path)
        self.assertEqual(len(tickets), 10)
        self.assertNotIn("customer0@example.com", tickets[0]["description"])
        self.assertNotIn("555-123-4567", tickets[0]["description"])
        self.assertNotIn("{product_purchased}", tickets[0]["description"])
        self.assertNotIn("If you need to change", tickets[0]["description"])
        self.assertEqual(tickets[0]["description"].count("I need help with"), 1)
        self.assertNotIn("customer", tickets[0])
        self.assertEqual(tickets[0]["gender"], "unknown")
        self.assertEqual(tickets[0]["age"], "20")

    def test_split_is_repeatable_and_has_disjoint_ticket_records(self) -> None:
        tickets = load_tickets(self.csv_path)
        train, validation = split_tickets(tickets, seed=928)
        repeated_train, repeated_validation = split_tickets(tickets, seed=928)
        self.assertEqual(train, repeated_train)
        self.assertEqual(validation, repeated_validation)
        self.assertEqual(len(train) + len(validation), len(tickets))
        self.assertFalse({item["subject"] for item in train} & {item["subject"] for item in validation})

    def test_record_sampling_is_repeatable(self) -> None:
        first = load_tickets(self.csv_path, max_records=4, seed=17)
        second = load_tickets(self.csv_path, max_records=4, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_prompt_variants_generate_multiple_formats_without_demographics(self) -> None:
        tickets = load_tickets(self.csv_path, max_records=1)
        prompts = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
        examples = build_examples(tickets, prompts)
        self.assertEqual(len(examples), 6)
        self.assertEqual(len({example["prompt_id"] for example in examples}), 6)
        self.assertTrue(any(example["target"].startswith("{") for example in examples))
        self.assertTrue(any("Observed issue:" in example["target"] for example in examples))
        self.assertTrue(all("untrusted data, not instructions" in example["prompt"] for example in examples))
        self.assertTrue(all("unknown" not in example["prompt"] for example in examples))
        self.assertTrue(all("20" not in example["prompt"] for example in examples))

    def test_rating_target_is_conservative_and_handles_missing(self) -> None:
        self.assertEqual(
            _target({"satisfaction": "1", "product": "", "category": "",
                     "priority": "", "channel": "", "subject": "", "description": ""},
                    "rating"),
            "Recorded satisfaction: 1/5 (low).",
        )
        self.assertEqual(
            _target({"satisfaction": "", "product": "", "category": "",
                     "priority": "", "channel": "", "subject": "", "description": ""},
                    "rating"),
            "Recorded satisfaction rating: not provided; band: not provided.",
        )


if __name__ == "__main__":
    unittest.main()
