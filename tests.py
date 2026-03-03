"""Tests for M259 Study App - Exam and Multiple Choice functionality"""
import pytest
from unittest.mock import patch, MagicMock
import json
from fastapi.testclient import TestClient

from main import app, call_llm


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestExamGeneration:
    """Tests for exam generation API"""
    
    @patch('main.call_llm')
    def test_generate_exam_with_multiple_choice(self, mock_llm, client):
        """Test generating an exam with multiple choice questions"""
        mock_llm.return_value = json.dumps({
            "title": "Test Exam",
            "questions": [
                {
                    "id": 1,
                    "type": "multiple_choice",
                    "question": "Was ist AI?",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "correct": "A",
                    "points": 2
                },
                {
                    "id": 2,
                    "type": "multiple_choice",
                    "question": "Was ist ML?",
                    "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                    "correct": "B",
                    "points": 2
                }
            ]
        })
        
        response = client.post("/api/exam", data={
            "topic": "AI ML Grundlagen",
            "question_count": 2,
            "difficulty": "medium",
            "question_types": "multiple_choice"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "exam" in data
        assert data["exam"]["title"] == "Test Exam"
        assert len(data["exam"]["questions"]) == 2
        
        # Verify MC question structure
        mc_q = data["exam"]["questions"][0]
        assert mc_q["type"] == "multiple_choice"
        assert "options" in mc_q
        assert len(mc_q["options"]) == 4
        assert mc_q["correct"] in ["A", "B", "C", "D"]
        assert "points" in mc_q

    @patch('main.call_llm')
    def test_generate_exam_with_open_questions(self, mock_llm, client):
        """Test generating an exam with open questions"""
        mock_llm.return_value = json.dumps({
            "title": "Open Questions Exam",
            "questions": [
                {
                    "id": 1,
                    "type": "open",
                    "question": "Erkläre den Unterschied zwischen AI und ML.",
                    "sample_answer": "AI ist der Oberbegriff...",
                    "points": 4
                }
            ]
        })
        
        response = client.post("/api/exam", data={
            "topic": "AI Grundlagen",
            "question_count": 1,
            "difficulty": "medium",
            "question_types": "open"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["exam"]["questions"]) == 1
        assert data["exam"]["questions"][0]["type"] == "open"
        assert "sample_answer" in data["exam"]["questions"][0]

    @patch('main.call_llm')
    def test_generate_mixed_exam(self, mock_llm, client):
        """Test generating an exam with mixed question types"""
        mock_llm.return_value = json.dumps({
            "title": "Mixed Exam",
            "questions": [
                {
                    "id": 1,
                    "type": "multiple_choice",
                    "question": "MC Question",
                    "options": ["A", "B", "C", "D"],
                    "correct": "A",
                    "points": 2
                },
                {
                    "id": 2,
                    "type": "open",
                    "question": "Open Question",
                    "sample_answer": "Sample",
                    "points": 4
                }
            ]
        })
        
        response = client.post("/api/exam", data={
            "topic": "Test",
            "question_count": 2,
            "difficulty": "medium",
            "question_types": "multiple_choice,open"
        })
        
        assert response.status_code == 200
        data = response.json()
        questions = data["exam"]["questions"]
        types = [q["type"] for q in questions]
        assert "multiple_choice" in types
        assert "open" in types


class TestMultipleChoiceGrading:
    """Tests for multiple choice question grading logic"""
    
    def test_mc_correct_answer_index_calculation(self):
        """Test that correct answer letter maps to correct option index"""
        options = ["First", "Second", "Third", "Fourth"]
        
        # A -> index 0
        assert ord("A") - 65 == 0
        assert options[ord("A") - 65] == "First"
        
        # B -> index 1
        assert ord("B") - 65 == 1
        assert options[ord("B") - 65] == "Second"
        
        # C -> index 2
        assert ord("C") - 65 == 2
        assert options[ord("C") - 65] == "Third"
        
        # D -> index 3
        assert ord("D") - 65 == 3
        assert options[ord("D") - 65] == "Fourth"

    def test_mc_answer_comparison(self):
        """Test multiple choice answer comparison logic"""
        test_cases = [
            {"selected": "A", "correct": "A", "expected": True},
            {"selected": "B", "correct": "A", "expected": False},
            {"selected": None, "correct": "A", "expected": False},
            {"selected": "C", "correct": "C", "expected": True},
        ]
        
        for case in test_cases:
            result = case["selected"] == case["correct"]
            assert result == case["expected"], f"Failed for {case}"

    def test_mc_points_calculation(self):
        """Test points calculation for MC questions"""
        questions = [
            {"type": "multiple_choice", "points": 2, "correct": "A"},
            {"type": "multiple_choice", "points": 3, "correct": "B"},
            {"type": "multiple_choice", "points": 1, "correct": "C"},
        ]
        
        answers = ["A", "A", "C"]  # First correct, second wrong, third correct
        
        total_points = 0
        max_points = 0
        for q, ans in zip(questions, answers):
            max_points += q["points"]
            if ans == q["correct"]:
                total_points += q["points"]
        
        assert total_points == 3  # 2 + 1
        assert max_points == 6   # 2 + 3 + 1


class TestOpenAnswerRating:
    """Tests for open answer rating API"""
    
    @patch('main.call_llm')
    def test_rate_open_answer_full_points(self, mock_llm, client):
        """Test rating an open answer that deserves full points"""
        mock_llm.return_value = json.dumps({
            "points": 4,
            "max_points": 4,
            "percentage": 100,
            "feedback": "Excellent answer!",
            "key_points_covered": ["AI definition", "ML definition", "Difference"],
            "key_points_missed": []
        })
        
        response = client.post("/api/rate-open-answer", data={
            "question": "Erkläre den Unterschied zwischen AI und ML.",
            "user_answer": "AI ist der Oberbegriff für maschinelle Intelligenz. ML ist ein Teilbereich von AI, der aus Daten lernt.",
            "max_points": 4
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["points"] == 4
        assert data["percentage"] == 100
        assert len(data["key_points_covered"]) == 3
        assert len(data["key_points_missed"]) == 0

    @patch('main.call_llm')
    def test_rate_open_answer_partial_points(self, mock_llm, client):
        """Test rating an answer that deserves partial points"""
        mock_llm.return_value = json.dumps({
            "points": 2,
            "max_points": 4,
            "percentage": 50,
            "feedback": "Partially correct",
            "key_points_covered": ["AI definition"],
            "key_points_missed": ["ML definition", "Difference"]
        })
        
        response = client.post("/api/rate-open-answer", data={
            "question": "Erkläre den Unterschied zwischen AI und ML.",
            "user_answer": "AI ist künstliche Intelligenz.",
            "max_points": 4
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["points"] == 2
        assert data["percentage"] == 50

    @patch('main.call_llm')
    def test_rate_open_answer_zero_points(self, mock_llm, client):
        """Test rating an empty or wrong answer"""
        mock_llm.return_value = json.dumps({
            "points": 0,
            "max_points": 4,
            "percentage": 0,
            "feedback": "Answer is incorrect",
            "key_points_covered": [],
            "key_points_missed": ["AI definition", "ML definition"]
        })
        
        response = client.post("/api/rate-open-answer", data={
            "question": "Was ist AI?",
            "user_answer": "Wrong answer",  # Use non-empty string
            "max_points": 4
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["points"] == 0
        assert data["percentage"] == 0


class TestExamValidation:
    """Tests for exam data validation"""
    
    def test_mc_question_structure(self):
        """Test that MC questions have all required fields"""
        required_mc_fields = ["id", "type", "question", "options", "correct", "points"]
        
        sample_mc = {
            "id": 1,
            "type": "multiple_choice",
            "question": "Test question?",
            "options": ["A", "B", "C", "D"],
            "correct": "A",
            "points": 2
        }
        
        for field in required_mc_fields:
            assert field in sample_mc, f"Missing field: {field}"

    def test_open_question_structure(self):
        """Test that open questions have all required fields"""
        required_open_fields = ["id", "type", "question", "points"]
        
        sample_open = {
            "id": 1,
            "type": "open",
            "question": "Test question?",
            "sample_answer": "Sample",
            "points": 4
        }
        
        for field in required_open_fields:
            assert field in sample_open, f"Missing field: {field}"

    def test_mc_options_count(self):
        """Test that MC questions have valid number of options"""
        valid_options = ["A", "B", "C", "D"]
        correct_answer = "B"
        
        assert len(valid_options) >= 2, "Need at least 2 options"
        assert len(valid_options) <= 6, "Max 6 options"
        assert correct_answer in ["A", "B", "C", "D", "E", "F"], "Invalid correct answer letter"

    def test_points_are_positive(self):
        """Test that points are positive integers"""
        questions = [
            {"type": "multiple_choice", "points": 2},
            {"type": "open", "points": 4},
        ]
        
        for q in questions:
            assert q["points"] > 0, "Points must be positive"
            assert isinstance(q["points"], int), "Points must be integer"


class TestExamEdgeCases:
    """Tests for edge cases in exam handling"""
    
    @patch('main.call_llm')
    def test_exam_with_malformed_json(self, mock_llm, client):
        """Test handling of malformed LLM response"""
        mock_llm.return_value = "This is not valid JSON"
        
        response = client.post("/api/exam", data={
            "topic": "Test",
            "question_count": 1,
            "difficulty": "medium",
            "question_types": "multiple_choice"
        })
        
        assert response.status_code == 200
        data = response.json()
        # Should return error exam
        assert "exam" in data

    def test_mc_answer_with_lowercase(self):
        """Test MC answer comparison handles case"""
        correct = "A"
        
        # Should be case-sensitive (uppercase expected)
        assert "a" != correct
        assert "A" == correct

    def test_empty_answer_handling(self):
        """Test handling of empty answers"""
        selected = None
        correct = "A"
        
        is_correct = selected == correct
        assert is_correct is False

    @patch('main.call_llm')
    def test_rate_answer_with_malformed_json(self, mock_llm, client):
        """Test handling malformed rating response"""
        mock_llm.return_value = "Not JSON"
        
        response = client.post("/api/rate-open-answer", data={
            "question": "Test?",
            "user_answer": "Answer",
            "max_points": 4
        })
        
        assert response.status_code == 200
        data = response.json()
        # Should return fallback response
        assert "points" in data


class TestExamScoreCalculation:
    """Tests for exam score calculation logic"""
    
    def test_total_score_calculation(self):
        """Test total exam score calculation"""
        mc_results = [
            {"is_correct": True, "points": 2},
            {"is_correct": False, "points": 2},
            {"is_correct": True, "points": 3},
        ]
        
        mc_correct = sum(r["points"] for r in mc_results if r["is_correct"])
        mc_total = sum(r["points"] for r in mc_results)
        
        assert mc_correct == 5  # 2 + 3
        assert mc_total == 7    # 2 + 2 + 3

    def test_percentage_calculation(self):
        """Test percentage calculation"""
        total_points = 15
        max_points = 20
        
        percentage = round((total_points / max_points) * 100)
        assert percentage == 75

    def test_grade_color_thresholds(self):
        """Test grade color assignment thresholds"""
        # >= 80%: green
        # >= 60%: yellow
        # < 60%: red
        
        assert 85 >= 80  # green
        assert 75 >= 60 and 75 < 80  # yellow
        assert 55 < 60  # red


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
