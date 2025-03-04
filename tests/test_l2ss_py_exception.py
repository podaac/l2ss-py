import pytest
import traceback
import sys
from podaac.subsetter.subset_harmony import L2SSException


def test_exception_message_formatting():
    """
    Test that the L2SSException correctly formats the error message
    with file, line, function, and original error details.
    """
    try:
        # Simulate an error by intentionally causing a division by zero
        1 / 0
    except ZeroDivisionError as original_error:
        l2ss_exception = L2SSException(original_error)
        
        # Detailed assertions with informative error messages
        error_str = str(l2ss_exception)
        error_msg = l2ss_exception.message
        
        assert "Error in file" in error_msg, f"Expected file context, got: {error_msg}"
        assert "line" in error_msg, f"Expected line number, got: {error_msg}"
        assert "in function" in error_msg, f"Expected function context, got: {error_msg}"
        assert "division by zero" in error_msg, f"Expected original error message, got: {error_msg}"
        assert l2ss_exception.category == 'podaac/l2-subsetter'

def test_exception_traceback_details():
    """
    Verify that the exception captures the correct traceback information.
    """
    def inner_function():
        # Another function to add depth to the traceback
        1 / 0
    
    try:
        inner_function()
    except ZeroDivisionError as original_error:
        l2ss_exception = L2SSException(original_error)
        
        # Extract expected details
        tb = traceback.extract_tb(original_error.__traceback__)[-1]
        expected_filename = tb.filename
        expected_lineno = tb.lineno
        expected_funcname = tb.name
        
        error_msg = l2ss_exception.message
        assert expected_filename in error_msg, f"Filename not found, got: {error_msg}"
        assert str(expected_lineno) in error_msg, f"Line number not found, got: {error_msg}"
        assert expected_funcname in error_msg, f"Function name not found, got: {error_msg}"

def test_original_error_type_preservation():
    """
    Ensure that the original error type is preserved in the traceback.
    """
    try:
        raise ValueError("Test error message")
    except ValueError as original_error:
        l2ss_exception = L2SSException(original_error)
        
        error_msg = l2ss_exception.message
        assert "Test error message" in error_msg, f"Original error message not found, got: {error_msg}"
        assert isinstance(l2ss_exception.original_exception, ValueError)

def test_module_identifier():
    """
    Verify that the module identifier is set correctly.
    """
    try:
        raise RuntimeError("Sample error")
    except RuntimeError as original_error:
        l2ss_exception = L2SSException(original_error)
        
        assert l2ss_exception.category == 'podaac/l2-subsetter'

def test_exception_with_no_traceback():
    """
    Test handling of an exception without an existing traceback.
    """
    # Create an exception without a traceback
    try:
        raise ValueError("Test exception without traceback")
    except ValueError as original_error:
        # Deliberately remove the traceback
        original_error.__traceback__ = None
        
        # Create L2SSException
        l2ss_exception = L2SSException(original_error)
        
        # Verify that a traceback was generated
        assert l2ss_exception.original_exception is not None
        
        # Check that the message is still formatted
        error_msg = l2ss_exception.message
        assert "Error in file" in error_msg
        assert "in function" in error_msg
        assert "Test exception without traceback" in error_msg
