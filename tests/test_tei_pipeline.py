import pytest
import os
import json
import shutil
import tempfile
from unittest.mock import patch
from scripts.tei_pipeline import write_output_to_multiple_files, process_directory

@pytest.fixture
def temp_data_dir():
    # Setup temporary directory for output
    temp_dir = tempfile.mkdtemp()
    
    # Note: write_output_to_multiple_files has hardcoded 'data' directory.
    # To test it cleanly without modifying the actual 'data' folder of the project, 
    # we temporarily change the working directory or mock the Path("data") creation.
    # A cleaner approach in testing is to just run in a temporary current working directory.
    
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    # Teardown
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)

def test_write_output_to_multiple_files_no_split(temp_data_dir):
    sample_objs = [
        {"id": "1", "training_text": "A" * 100},
        {"id": "2", "training_text": "B" * 100}
    ]
    
    # Call with a large max_size_mb so it doesn't split
    write_output_to_multiple_files(sample_objs, corpus_id="test_corpus", max_size_mb=10)
    
    # Check if data directory was created
    assert os.path.exists("data")
    
    # Check if only one file was created
    files = os.listdir("data")
    assert len(files) == 1
    assert files[0] == "test_corpus_0.json"
    
    # Check contents
    with open(os.path.join("data", files[0]), "r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["id"] == "1"

def test_write_output_to_multiple_files_with_split(temp_data_dir):
    sample_objs = [
        {"id": str(i), "training_text": "A" * 500} for i in range(10)
    ]
    
    # 500 bytes per object approx. 
    # Let's set max_size_mb to a very very small fraction to force split
    # max_size_mb is in MB. So 1 / 1024 / 1024 = 1 byte. 
    # Let's use 0.001 MB which is approx 1048 bytes.
    # Each object is ~550 bytes when json dumped. So 2 objects per file.
    
    write_output_to_multiple_files(sample_objs, corpus_id="test_split", max_size_mb=0.001)
    
    files = os.listdir("data")
    # if each file contains max 2 objects, we should have around 5 files.
    assert len(files) > 1
    
    total_objects_loaded = 0
    for file in files:
        assert file.startswith("test_split_")
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            data = json.load(f)
            total_objects_loaded += len(data)
            
    assert total_objects_loaded == 10

@patch('scripts.tei_pipeline.convert_tei_to_json')
def test_process_directory(mock_convert, temp_data_dir):
    # Setup mock behavior
    mock_convert.return_value = [
        {"corpus_id": "test_corpus", "training_text": "mocked text"}
    ]
    
    # Create fake xml files
    os.makedirs("input_dir")
    with open("input_dir/a.xml", "w") as f:
        f.write("<test/>")
    with open("input_dir/b.txt", "w") as f:
        f.write("ignore me")
        
    process_directory("input_dir")
    
    # convert_tei_to_json should be called only for .xml files
    assert mock_convert.call_count == 1
    
    # Verify outputs
    assert os.path.exists("data")
    files = os.listdir("data")
    assert len(files) == 1
    assert files[0] == "test_corpus_0.json"
