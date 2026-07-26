import io
import unittest
from unittest.mock import patch
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import shutil

FILEPATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'structured_restaurant_data.json')
BACKUP_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'structured_restaurant_data.json.bak')
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
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_data(data, file_path, backup_path):
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)
    with open(file_path, 'w') as f:
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
    return response.choices[0].message.content


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

    while True:
        try:
            restaurant_data = Restaurant.model_validate_json(candidate_json_output)
            break
        except ValidationError as e:
            error_message = e.json()
            auto_repair_system_msg, auto_repair_prompt = JSON_auto_repair_prompts(
                candidate_json_output, error_message
            )
            candidate_json_output = llm_model(
                system_msg=auto_repair_system_msg,
                prompt_txt=auto_repair_prompt,
            )

    result = restaurant_data.model_dump()
    result["itemId"] = itemId
    return result

def manage_restaurants(file_path, backup_path):
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        if choice == '1':
            print("\n--- Current Listings ---")
            for record in data:
                print(record.get('name', 'N/A'))

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
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != 'yes':
                print("Operation cancelled.")
                continue

            if choice == '3':
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter the new restaurant description: ")
                new_record = new_data_entry_process(paragraph, itemId)
                data.append(new_record)
                save_data(data, file_path, backup_path)
                print("✅ Restaurant added.")

            elif choice == '4':
                try:
                    index = int(input("Enter record index: "))
                except ValueError:
                    print("invalid index.")
                    continue

                if 0 <= index < len(data):
                    record = data[index]
                    for key in record:
                        new_val = input(f"New value for '{key}' (Enter to skip): ")
                        if new_val.strip():
                            record[key] = new_val
                    save_data(data, file_path, backup_path)
                    print("✅ Record updated.")
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
                    print("✅ Record deleted.")
                else:
                    print("invalid index.")

        elif choice == '6':
            break
        else:
            print("Invalid input.")

class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary clean database for testing."""
        self.test_file = 'structured_restaurant_data_unit_test.json'
        self.test_file_backup = 'structured_restaurant_data_unit_test.json.bak'
        self.initial_data = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, 'w') as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        """Clean up the test file after tests."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant.
        Inputs: '3' (Add), 'yes' (Confirm), 'New Burger Joint', '6' (Exit)
        """
        mock_restaurant = 'The Copper Sprout is a high-concept, Modern Appalachian farm-to-table destination that blends an industrial-chic aesthetic with rustic forest charm, featuring reclaimed wood and amber lighting to create a sophisticated yet cozy vibe. Priced in the $$ category, the menu celebrates seasonal foraging and local heritage, headlined by signature dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience is designed to be intimate and earthy, making it a premier spot for those seeking high-quality, smokehouse-influenced cuisine in a refined, atmospheric setting.'
        mock_input.side_effect = ['3', 'yes', mock_restaurant, '6']

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        mock_input.side_effect = ['5', 'yes', 1, '6']

        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        print(data)
        self.assertEqual(len(data), 1)

    @patch('builtins.input')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but say 'no' to security warning.
        Inputs: '5' (Delete), 'no' (Cancel), '6' (Exit)
        """
        mock_input.side_effect = ['5', 'no', '6']

        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
    # manage_restaurants(FILEPATH, BACKUP_PATH)

