from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder
from typing import Dict, List
import json
import jsonschema

class SchemaInferenceAgent:
    def __init__(self, openai_api_key: str):
        # Initialize the LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            openai_api_key=openai_api_key
        )
        
        # Create tools for the agent
        self.tools = [
            Tool(
                name="infer_schema",
                func=self._infer_json_schema,
                description="Infer JSON schema from a given JSON data example"
            ),
            Tool(
                name="validate_schema",
                func=self._validate_schema,
                description="Validate if a JSON schema correctly matches the given data"
            )
        ]
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs={
                "system_message": SystemMessage(content="""You are an expert at JSON Schema inference and validation. 
                Your task is to analyze JSON data and create accurate JSON schemas that properly validate the data structure.""")
            }
        )

    def _get_json_type(self, python_type: str) -> str:
        """Convert Python type names to JSON schema types"""
        type_mapping = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'NoneType': 'null',
            'list': 'array',
            'dict': 'object'
        }
        return type_mapping.get(python_type, 'string')

    def _is_date_format(self, value: str) -> bool:
        """Check if string matches common date formats"""
        import re
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)

    def _infer_json_schema(self, json_data: str) -> str:
        """Infer JSON schema from example data"""
        try:
            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            
            def infer_type(value, field_name=""):
                if isinstance(value, dict):
                    properties = {}
                    required = []
                    for key, val in value.items():
                        properties[key] = infer_type(val, key)
                        required.append(key)
                    return {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                elif isinstance(value, list):
                    if not value:
                        return {"type": "array", "items": {}}
                    
                    # Check if array has mixed types
                    types = {type(item).__name__ for item in value}
                    if len(types) > 1:
                        return {
                            "type": "array",
                            "items": {
                                "oneOf": [
                                    {"type": self._get_json_type(type(item).__name__)}
                                    for item in value
                                ]
                            }
                        }
                    
                    item_schema = infer_type(value[0])
                    
                    # Add array constraints for numbers array
                    if all(isinstance(x, (int, float)) for x in value):
                        return {
                            "type": "array",
                            "items": item_schema,
                            "minItems": 1,
                            "maxItems": 10,
                            "uniqueItems": len(value) == len(set(value))
                        }
                    return {"type": "array", "items": item_schema}
                elif isinstance(value, str):
                    # Infer string formats
                    if field_name.lower() == "email" or "@" in value:
                        return {"type": "string", "format": "email"}
                    elif field_name.lower() == "date" or self._is_date_format(value):
                        return {"type": "string", "format": "date"}
                    return {"type": "string"}
                elif isinstance(value, bool):
                    return {"type": "boolean"}
                elif isinstance(value, int):
                    return {"type": "integer"}
                elif isinstance(value, float):
                    return {"type": "number"}
                elif value is None:
                    return {"type": "null"}
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")

            schema = infer_type(data)
            schema["$schema"] = "http://json-schema.org/draft-07/schema#"
            return json.dumps(schema, indent=2)
        except Exception as e:
            return f"Error inferring schema: {str(e)}"

    def _validate_schema(self, data: Dict) -> bool:
        """Validate JSON data against its schema"""
        try:
            schema = data.get("schema")
            instance = data.get("data")
            if not schema or not instance:
                return False
            jsonschema.validate(instance=instance, schema=schema)
            return True
        except jsonschema.exceptions.ValidationError:
            return False
        except Exception as e:
            return f"Error validating schema: {str(e)}"

    def _compare_schemas(self, inferred_schema: dict, expected_schema: dict) -> Dict:
        """Compare inferred schema with expected schema and return detailed results"""
        differences = []
        
        def compare_recursive(inferred: dict, expected: dict, path: str = ""):
            if isinstance(inferred, dict) and isinstance(expected, dict):
                # Compare types
                if inferred.get("type") != expected.get("type"):
                    differences.append({
                        "path": path,
                        "issue": "type_mismatch",
                        "inferred": inferred.get("type"),
                        "expected": expected.get("type")
                    })
                
                # Compare string formats
                if expected.get("type") == "string" and "format" in expected:
                    if "format" not in inferred:
                        differences.append({
                            "path": path,
                            "issue": "missing_format",
                            "expected_format": expected["format"]
                        })
                    elif inferred["format"] != expected["format"]:
                        differences.append({
                            "path": path,
                            "issue": "format_mismatch",
                            "inferred": inferred["format"],
                            "expected": expected["format"]
                        })
                
                # Compare array constraints
                if expected.get("type") == "array":
                    for constraint in ["minItems", "maxItems", "uniqueItems"]:
                        if constraint in expected and constraint not in inferred:
                            differences.append({
                                "path": path,
                                "issue": "missing_array_constraint",
                                "constraint": constraint,
                                "expected": expected[constraint]
                            })
                
                # Compare required fields
                inferred_required = set(inferred.get("required", []))
                expected_required = set(expected.get("required", []))
                if inferred_required != expected_required:
                    differences.append({
                        "path": path,
                        "issue": "required_fields_mismatch",
                        "missing": list(expected_required - inferred_required),
                        "extra": list(inferred_required - expected_required)
                    })
                
                # Compare properties recursively
                inferred_props = inferred.get("properties", {})
                expected_props = expected.get("properties", {})
                for prop in set(inferred_props.keys()) | set(expected_props.keys()):
                    new_path = f"{path}.{prop}" if path else prop
                    if prop in inferred_props and prop in expected_props:
                        compare_recursive(inferred_props[prop], expected_props[prop], new_path)
                    elif prop not in inferred_props:
                        differences.append({
                            "path": new_path,
                            "issue": "missing_property",
                            "expected": expected_props[prop]
                        })
                    else:
                        differences.append({
                            "path": new_path,
                            "issue": "extra_property",
                            "inferred": inferred_props[prop]
                        })
                
                # Compare array item types
                if expected.get("type") == "array" and "items" in expected:
                    if "oneOf" in expected["items"]:
                        if "oneOf" not in inferred.get("items", {}):
                            differences.append({
                                "path": f"{path}.items",
                                "issue": "missing_mixed_types",
                                "expected": expected["items"]["oneOf"]
                            })
        
        compare_recursive(inferred_schema, expected_schema)
        return {
            "passed": len(differences) == 0,
            "differences": differences
        }

    def process_test_cases(self, test_cases_file: str) -> List[Dict]:
        """Process test cases from a JSON file and compare with expected schemas"""
        with open(test_cases_file, 'r') as f:
            test_cases = json.load(f)

        results = []
        for test_case in test_cases:
            result = {
                "test_case_id": test_case.get("id"),
                "input_data": test_case.get("input_data"),
            }

            # Infer schema
            inferred_schema = self._infer_json_schema(test_case["input_data"])
            result["inferred_schema"] = json.loads(inferred_schema)

            # Validate schema
            validation_result = self._validate_schema({
                "schema": result["inferred_schema"],
                "data": test_case["input_data"]
            })
            result["validation_success"] = validation_result

            # Compare with expected schema
            if "expected_schema" in test_case:
                comparison_result = self._compare_schemas(
                    result["inferred_schema"],
                    test_case["expected_schema"]
                )
                result["schema_comparison"] = comparison_result
                result["test_passed"] = comparison_result["passed"]
            
            results.append(result)

        return results 