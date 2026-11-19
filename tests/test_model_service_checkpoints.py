from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pickle
import pytest
import zlib

from backend.api.exceptions import DatasetNotFoundError, GreekSchoolsError
from backend.api.services.model_service import ModelService
from backend.config.settings import (
    NGRAM_CHECKPOINTS_DIR,
    get_ngram_checkpoint_path,
)


def test_get_ngram_checkpoint_path():
    mle_path = get_ngram_checkpoint_path(lm_type="MLE", n=3)
    assert mle_path == NGRAM_CHECKPOINTS_DIR / "ngram_mle_n3.pkl.zlib"

    lidstone_path = get_ngram_checkpoint_path(lm_type="LIDSTONE", n=3, gamma=0.001)
    assert lidstone_path == NGRAM_CHECKPOINTS_DIR / "ngram_lidstone_n3_gamma0.001.pkl.zlib"

    custom_dir = Path("/tmp/custom_ckpts")
    custom_path = get_ngram_checkpoint_path(
        lm_type="LIDSTONE", n=2, gamma=0.05, base_dir=custom_dir
    )
    assert custom_path == custom_dir / "ngram_lidstone_n2_gamma0.05.pkl.zlib"


def test_dataset_not_found_error_hierarchy():
    assert issubclass(DatasetNotFoundError, GreekSchoolsError)
    err = DatasetNotFoundError("test message")
    assert str(err) == "test message"


@pytest.mark.asyncio
async def test_model_service_loads_checkpoint(tmp_path):
    # Creiamo un finto checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_file = ckpt_dir / "ngram_mle_n3.pkl.zlib"

    fake_payload = {
        "lm_type": "MLE",
        "gamma": None,
        "n": 3,
        "global_model": "fake_global_model",
        "domain_model": "fake_domain_model",
    }
    ckpt_file.write_bytes(zlib.compress(pickle.dumps(fake_payload)))

    mock_collection = MagicMock()
    mock_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id="fake_mongo_id")
    )
    mock_fs = MagicMock()
    mock_fs.upload_from_stream = AsyncMock()

    service = ModelService(db_collection=mock_collection, gridfs=mock_fs)

    with patch(
        "backend.api.services.model_service.get_ngram_checkpoint_path",
        return_value=ckpt_file,
    ), patch("backend.api.services.model_service.pipeline_train") as mock_train:
        model_dict = {
            "LM_SCORE": "MLE",
            "N": 3,
            "GAMMA": None,
        }
        res_id = await service._train_and_persist_ngram(model_dict)
        assert res_id == "fake_mongo_id"
        mock_train.assert_not_called()
        assert mock_fs.upload_from_stream.call_count == 2
        assert model_dict["GLOBAL_MODEL_FILE_ID"] is not None
        assert model_dict["DOMAIN_MODEL_FILE_ID"] is not None


@pytest.mark.asyncio
async def test_model_service_raises_when_no_checkpoint_and_no_data(tmp_path):
    empty_data_dir = tmp_path / "non_existent_data"
    non_existent_ckpt = tmp_path / "checkpoints" / "ngram_mle_n3.pkl.zlib"

    mock_collection = MagicMock()
    mock_fs = MagicMock()
    service = ModelService(db_collection=mock_collection, gridfs=mock_fs)

    with patch(
        "backend.api.services.model_service.get_ngram_checkpoint_path",
        return_value=non_existent_ckpt,
    ), patch(
        "backend.api.services.model_service.DATA_PATH",
        empty_data_dir,
    ):
        model_dict = {
            "LM_SCORE": "MLE",
            "N": 3,
            "GAMMA": None,
        }
        with pytest.raises(DatasetNotFoundError) as exc_info:
            await service._train_and_persist_ngram(model_dict)

        assert "Pretrained checkpoint for N-gram model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_init_models_registers_ithaca_and_featured_rag_checkpoints():
    mock_collection = MagicMock()
    mock_collection.find_one = AsyncMock(return_value=None)
    inserted_counter = 0

    async def _fake_insert(doc):
        nonlocal inserted_counter
        inserted_counter += 1
        return MagicMock(inserted_id=f"id-{inserted_counter}")

    mock_collection.insert_one = AsyncMock(side_effect=_fake_insert)
    mock_fs = MagicMock()

    service = ModelService(db_collection=mock_collection, gridfs=mock_fs)

    with patch.object(
        service, "_create_ngram_model_from_params", new=AsyncMock(return_value="ngram-id")
    ), patch.object(
        service, "_create_bert_model_from_checkpoint", new=AsyncMock(return_value="bert-id")
    ):
        ids = await service.init_models()

    inserted_docs = [call.args[0] for call in mock_collection.insert_one.call_args_list]
    assert any(
        d.get("TYPE") == "Ithaca" and d.get("CHECKPOINT") == "CNR-ILC/gs-ithaca-tlg"
        for d in inserted_docs
    )
    assert any(
        d.get("TYPE") == "RAG" and d.get("CHECKPOINT") == "Qwen/Qwen2.5-72B-Instruct"
        for d in inserted_docs
    )
    assert any(
        d.get("TYPE") == "RAG" and d.get("CHECKPOINT") == "meta-llama/Llama-3.3-70B-Instruct"
        for d in inserted_docs
    )
    assert any(
        d.get("TYPE") == "RAG"
        and d.get("CHECKPOINT") == "CNR-ILC/gs-GreBerta"
        and d.get("BACKEND") == "neuro_lexical"
        for d in inserted_docs
    )
    assert len(ids) > 0
