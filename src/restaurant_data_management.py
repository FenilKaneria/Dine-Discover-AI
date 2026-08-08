import argparse
import io
import json
import logging
import os
import shutil
import sys
import unittest
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Load .env from the project root before anything reads os.environ
logging.getLogger("dotenv.main").setLevel(logging.ERROR)
load_dotenv(Path(__file__).parent.parent / ".env")

FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'structured_restaurant_data.json')
BACKUP_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'structured_restaurant_data.json.bak')

# The repair loop must terminate — a model that keeps returning bad JSON would
# otherwise spin forever (and keep billing).
MAX_REPAIR_ATTEMPTS = 3

EXAMPLE_RESTAURANT_PARAGRAPH = (
    'Down in **Santa Monica**, **Mar de Cortez** serves as a **sun-drenched**, '
    '**casual taqueria** specializing in **Baja-style seafood**. With a **4.2/5** '
    'rating, it captures the salt-air energy of the coast through its signature '
    'beer-battered snapper tacos and zesty octopus ceviche, making it a premier '
    'spot for open-air dining near the pier. Price range: $'
)
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
    }}
"""


class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


def load_data(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # A corrupted file that parses to a non-list would break every caller.
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return []
    return []


def save_data(data, file_path, backup_path):
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def show_restaurant_card(data, index):
    res = data[index]
    print(f"\n{'=' * 44}")
    print(f"  {res.get('name', 'N/A')}")
    print(f"{'=' * 44}")
    for key, value in res.items():
        if key != 'name':
            label = key.replace('_', ' ').title()
            print(f"  {label}: {value}")


def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
    You are a precise information extraction assistant.
    Extract restaurant attributes from the provided description and return only a valid JSON object.
    Follow this schema exactly:
    {{
      "name": string,
      "location": string,
      "type": string,
      "food_style": string,
      "rating": number or null,
      "price_range": integer or null,
      "signatures": array of strings,
      "vibe": string or null,
      "environment": string,
      "shortcomings": array of strings
    }}
    Rules:
    - Return JSON only. No markdown, no code fences, no explanation.
    - Do not invent facts. Use only information stated or clearly implied in the description.
    - Convert price symbols like $, $$, $$$, or $$$$ into the corresponding integer count.
    - Use empty arrays for signatures or shortcomings if none are mentioned.
    - Keep the output strictly valid JSON.
    """

    base_user_prompt = f"""
    Task: Convert the following restaurant description into structured JSON using the schema and rules provided.

    Restaurant description:
    {restaurant_paragraph}

    Example:
    Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
    Output:
    {EXAMPLE_OUTPUT}

    """
    return base_system_msg, base_user_prompt


def llm_model(system_msg, prompt_txt, params=None):
    # Imported and constructed lazily so this module can be imported (and its
    # tests run) without an API key present.
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_BASE"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file in the project root "
            "before adding restaurants (this step calls the LLM)."
        )

    client = OpenAI()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt},
    ]

    response = client.chat.completions.create(
        model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        messages=messages,
        temperature=params.get("temperature", 0.7) if params else 0.7
    )
    return response.choices[0].message.content or ""


def strip_code_fences(text):
    """Models often wrap JSON in ```json fences despite being told not to."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]                      # drop the opening ``` / ```json
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]                 # drop the closing fence
        text = "\n".join(lines).strip()
    return text


def JSON_auto_repair_prompts(response, error_message):
    auto_repair_system_msg = f"""
        You are a JSON repair assistant.
        Your task is to fix invalid or non-conforming restaurant JSON so it matches the required schema exactly.

        Rules:
        - Return JSON only. No markdown, no explanation, no code fences.
        - Preserve all valid information from the candidate output.
        - Use the error message to correct schema, type, and formatting issues.
        - Ensure the final output is valid JSON and conforms to this structure:
        {{
          "name": string,
          "location": string,
          "type": string,
          "food_style": string,
          "rating": number or null,
          "price_range": integer or null,
          "signatures": array of strings,
          "vibe": string or null,
          "environment": string,
          "shortcomings": array of strings
        }}
        - If a field is missing and cannot be recovered, use null for optional scalar fields and empty arrays for list fields.
        - Do not add any new facts that are not supported by the candidate output.
    """
    auto_repair_prompt = f"""
        The following JSON output failed validation.

        Candidate JSON output:
        {response}

        Validation error:
        {error_message}

        Fix the JSON so it conforms exactly to the required schema and return only the repaired JSON object.
    """
    return auto_repair_system_msg, auto_repair_prompt


def new_data_entry_process(paragraph, itemId):
    base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(
        restaurant_paragraph=paragraph
    )
    candidate_json_output = llm_model(
        system_msg=base_system_msg,
        prompt_txt=base_user_prompt,
    )

    last_error = None
    for _ in range(MAX_REPAIR_ATTEMPTS):
        try:
            restaurant_data = Restaurant.model_validate_json(
                strip_code_fences(candidate_json_output)
            )
            result = restaurant_data.model_dump()
            result["itemId"] = itemId
            return result
        except ValidationError as e:
            last_error = e
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(
                candidate_json_output, e.json()
            )
            candidate_json_output = llm_model(
                system_msg=auto_repair_system_msg,
                prompt_txt=auto_repair_prompt,
            )

    raise ValueError(
        f"Could not produce valid restaurant JSON after {MAX_REPAIR_ATTEMPTS} attempts. "
        f"Last validation error: {last_error}"
    )


def coerce_to_field_type(existing_value, new_value):
    """Keep the stored type when editing, so a rating stays a number and a list
    stays a list instead of silently becoming a string."""
    new_value = new_value.strip()
    if isinstance(existing_value, bool):
        return new_value.lower() in ("true", "yes", "1")
    if isinstance(existing_value, int):
        return int(float(new_value))
    if isinstance(existing_value, float):
        return float(new_value)
    if isinstance(existing_value, list):
        return [part.strip() for part in new_value.split(",") if part.strip()]
    return new_value


def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\nRESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        try:
            choice = input("\nAction: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C / piped input ending should exit cleanly, not traceback.
            print("\nExiting.")
            return

        if choice == '1':
            print("\n--- Current Listings ---")
            for i, record in enumerate(data):
                print(f"  [{i}] {record.get('name', 'N/A')}")

        elif choice == '2':
            try:
                index = int(input("Enter record index: "))
            except ValueError:
                print("invalid index.")
                continue

            if 0 <= index < len(data):
                show_restaurant_card(data, index)
            else:
                print("invalid index.")

        elif choice in ['3', '4', '5']:
            print("\nSECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter the new restaurant description: ")
                if not paragraph.strip():
                    print("No description provided. Operation cancelled.")
                    continue
                try:
                    new_record = new_data_entry_process(paragraph, itemId)
                except Exception as exc:
                    # A failed LLM call must not take down the whole session.
                    print(f"Could not add restaurant: {type(exc).__name__}: {exc}")
                    continue
                data.append(new_record)
                save_data(data, file_path, backup_path)
                print("Restaurant added.")

            elif choice == '4':
                try:
                    index = int(input("Enter record index: "))
                except ValueError:
                    print("invalid index.")
                    continue

                if 0 <= index < len(data):
                    record = data[index]
                    for key in list(record):
                        new_val = input(f"New value for '{key}' (Enter to skip): ")
                        if new_val.strip():
                            try:
                                record[key] = coerce_to_field_type(record[key], new_val)
                            except ValueError:
                                print(f"  Skipped '{key}': '{new_val}' is not a valid {type(record[key]).__name__}.")
                    save_data(data, file_path, backup_path)
                    print("Record updated.")
                else:
                    print("invalid index.")

            elif choice == '5':
                try:
                    index = int(input("Enter record index: "))
                except ValueError:
                    print("invalid index.")
                    continue

                if 0 <= index < len(data):
                    data.pop(index)
                    save_data(data, file_path, backup_path)
                    print("Record deleted.")
                else:
                    print("invalid index.")

        elif choice == '6':
            break
        else:
            print("Invalid input.")


VALID_LLM_JSON = json.dumps({
    "name": "The Copper Sprout",
    "location": "Unknown",
    "type": "farm-to-table destination",
    "food_style": "Modern Appalachian",
    "rating": None,
    "price_range": 2,
    "signatures": ["Cast-Iron Smoked Trout", "Wild Mushroom Risotto"],
    "vibe": "industrial-chic with rustic forest charm",
    "environment": "reclaimed wood and amber lighting, intimate and earthy",
    "shortcomings": [],
})


class TestRestaurantDatabase(unittest.TestCase):
    """These tests never call the network — llm_model is mocked, so the suite is
    deterministic, free, and safe to run during a live demo."""

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w', encoding='utf-8') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        for path in (self.test_file, self.test_file_backup):
            if os.path.exists(path):
                os.remove(path)

    def read_test_file(self):
        with open(self.test_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    @patch(f'{__name__}.llm_model', return_value=VALID_LLM_JSON)
    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input, mock_llm):
        """
        Test Scenario: Add a new restaurant, then delete it.
        Inputs: '3' (Add), 'yes' (Confirm), <description>, '6' (Exit)
        """
        mock_restaurant = (
            'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table '
            'destination that blends an industrial-chic aesthetic with rustic forest charm.'
        )
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']
        manage_restaurants(self.test_file, self.test_file_backup)

        data = self.read_test_file()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[1]['name'], 'The Copper Sprout')
        self.assertEqual(data[1]['itemId'], 1000002)
        self.assertIn("Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', '1', '6']
        manage_restaurants(self.test_file, self.test_file_backup)

        data = self.read_test_file()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], 'Test Cafe')

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']
        manage_restaurants(self.test_file, self.test_file_backup)

        self.assertEqual(len(self.read_test_file()), 1)
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_invalid_index_does_not_crash(self, mock_stdout, mock_input):
        """Non-numeric and out-of-range indexes are rejected, not raised."""
        mock_input.side_effect = ['2', 'abc', '2', '99', '6']
        manage_restaurants(self.test_file, self.test_file_backup)
        self.assertEqual(mock_stdout.getvalue().count("invalid index."), 2)

    @patch(f'{__name__}.llm_model')
    def test_json_repair_loop_recovers(self, mock_llm):
        """A malformed first response is repaired on the retry, not looped forever."""
        mock_llm.side_effect = ["not json at all", f"```json\n{VALID_LLM_JSON}\n```"]
        record = new_data_entry_process("some description", 1234)
        self.assertEqual(record['name'], 'The Copper Sprout')
        self.assertEqual(record['itemId'], 1234)
        self.assertEqual(mock_llm.call_count, 2)

    @patch(f'{__name__}.llm_model', return_value="still not json")
    def test_json_repair_loop_gives_up(self, mock_llm):
        """The repair loop terminates instead of hanging on a broken model."""
        with self.assertRaises(ValueError):
            new_data_entry_process("some description", 1234)
        self.assertEqual(mock_llm.call_count, MAX_REPAIR_ATTEMPTS + 1)

    def test_coerce_preserves_field_types(self):
        self.assertEqual(coerce_to_field_type(4.5, "4.8"), 4.8)
        self.assertEqual(coerce_to_field_type(2, "3"), 3)
        self.assertEqual(coerce_to_field_type(["a"], "x, y"), ["x", "y"])
        self.assertEqual(coerce_to_field_type("old", "new"), "new")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the structured restaurant database.")
    parser.add_argument(
        "--test", action="store_true",
        help="Run the offline unit tests instead of the interactive manager.",
    )
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=[sys.argv[0]], verbosity=2)
    else:
        manage_restaurants(FILEPATH, BACKUP_PATH)
