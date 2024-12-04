from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from matmalab_app.main import app, generate_math_problem, get_db
from matmalab_app.tables.questions import Base, MathProblemInDB

client = TestClient(app)
HTTP_OK = 200
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def test_db():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def _override_get_db(test_db) -> None:
    app.dependency_overrides[get_db] = lambda: test_db
    yield
    del app.dependency_overrides[get_db]


@patch("matmalab_app.main.requests.post")
@patch("matmalab_app.main.pull_model")
def test_generate_math_problem(pull_model_mock, post_mock):
    pull_model_mock.return_value = None

    post_mock.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "response": '{"question": "What is 5 plus 3?", "answer": "8"}',
        },
    )

    result = generate_math_problem()

    assert result == '{"question": "What is 5 plus 3?", "answer": "8"}'
    pull_model_mock.assert_called_once_with("mathstral")
    post_mock.assert_called_once_with(
        "http://ollama:11434/api/generate",
        json={
            "model": "mathstral",
            "prompt": (
                "Generate a basic math problem and return it in a JSON format with these keys: 'question' and 'answer'.\n"
                "    The 'question' should describe the problem, and 'answer' should be the solution. For example:\n"
                "{\n"
                '  "question": "What is 9 divided by 3?",\n'
                '  "answer": "3"\n'
                "}\n"
            ),
            "format": "json",
            "stream": False,
        },
        timeout=60,
    )


@pytest.mark.usefixtures("_override_get_db")
@patch("matmalab_app.main.generate_math_problem")
def test_add_question(mock_generate_math_problem, test_db):

    mock_generate_math_problem.return_value = (
        '{"question": "What is 5 + 3?", "answer": "8"}'
    )

    response = client.post("/matmalab")

    assert response.status_code == HTTP_OK
    response_data = response.json()

    assert "question" in response_data
    assert response_data["question"] == "What is 5 + 3?"
    assert response_data["answer"] == "8"

    db_data = test_db.query(MathProblemInDB).all()
    assert len(db_data) == 1
    assert db_data[0].question == "What is 5 + 3?"
    assert db_data[0].answer == "8"


@pytest.mark.usefixtures("_override_get_db")
def test_fetch_questions(test_db):
    test_db.add(MathProblemInDB(question="What is 5 + 3?", answer="8"))
    test_db.commit()

    response = client.get("/matmalab")

    assert response.status_code == HTTP_OK
    response_data = response.json()
    assert len(response_data) == 1
    assert response_data[0]["question"] == "What is 5 + 3?"
    assert response_data[0]["answer"] == "8"


@pytest.mark.usefixtures("_override_get_db")
def test_delete_question(test_db):

    test_db.add(MathProblemInDB(question="What is 5 + 3?", answer="8"))
    test_db.commit()

    response = client.delete("/matmalab")
    assert response.status_code == HTTP_OK

    db_data = test_db.query(MathProblemInDB).all()
    assert len(db_data) == 0
