import pytest
from guardrail import validate_mapping

def test_happy_path_finance():
    """Test valid mappings with standard financial terms."""
    equation = r"P = V * \sigma"
    mapping = {
        "P": "Price",
        "V": "Volume",
        "\\sigma": "Volatility"
    }
    valid, msg = validate_mapping(equation, mapping)
    assert valid is True
    assert "Mapping is dimensionally consistent" in msg

def test_happy_path_single_letters():
    """Test valid mappings with single letter valid terms."""
    equation = r"P = V"
    mapping = {
        "P": "P",
        "V": "V"
    }
    valid, msg = validate_mapping(equation, mapping)
    assert valid is True

def test_missing_source_variable():
    """Test when a source variable is not in the equation."""
    equation = r"P = V"
    mapping = {
        "X": "Price" # X is not in equation
    }
    valid, msg = validate_mapping(equation, mapping)
    assert valid is False
    assert "Hallucination detected" in msg
    assert "X" in msg

def test_invalid_target_variable():
    """
    Test when a target variable does not contain any valid terms.
    'Z' does not contain P, V, L, t, N, r, K, H, O, E or any other valid term.
    """
    equation = r"P = Z"
    mapping = {
        "P": "Price",
        "Z": "Z" # Invalid target
    }
    valid, msg = validate_mapping(equation, mapping)
    assert valid is False
    assert "Invalid Target Terms" in msg
    assert "Z" in msg

def test_spacing_handling():
    """Test that validate_mapping correctly handles spacing differences."""
    # Equation has space inside \tilde {x}, source key is \tilde{x}
    equation = r"\tilde {x} = y"
    mapping = {
        r"\tilde{x}": "Price",
        "y": "Volume"
    }
    valid, msg = validate_mapping(equation, mapping)
    assert valid is True

    # Reverse case: Equation has \tilde{x}, source key is \tilde {x}
    # Note: validate_mapping normalizes equation but not keys?
    # Let's check code:
    # eq_norm = equation_tex.replace(" ", "")
    # if src_var not in equation_tex and src_var.replace(" ", "") not in eq_norm:
    # So if src_var has space "\tilde {x}", src_var.replace(" ", "") is "\tilde{x}".
    # eq_norm has "\tilde{x}". So it should match.

    equation2 = r"\tilde{x} = y"
    mapping2 = {
        r"\tilde {x}": "Price",
        "y": "Volume"
    }
    valid, msg = validate_mapping(equation2, mapping2)
    assert valid is True

def test_case_insensitivity():
    """Test that target matching is case-insensitive."""
    equation = r"P = V"
    mapping = {
        "P": "price", # lowercase 'price' should match 'Price'
        "V": "volume"
    }
    valid, msg = validate_mapping(equation, mapping)
    assert valid is True

def test_edge_cases_empty():
    """Test empty mapping and equation."""
    # Empty mapping should pass (vacuously true?)
    valid, msg = validate_mapping("P=V", {})
    assert valid is True # No invalid keys, no missing keys.

    # Empty equation with non-empty mapping should fail existence check
    valid, msg = validate_mapping("", {"P": "Price"})
    assert valid is False
    assert "Hallucination detected" in msg

def test_partial_match_target():
    """Test that partial matches work (as per implementation)."""
    # "MyPrice" contains "Price"
    equation = r"P = V"
    mapping = {
        "P": "MyPrice",
        "V": "SuperVolume"
    }
    valid, msg = validate_mapping(equation, mapping)
    assert valid is True
