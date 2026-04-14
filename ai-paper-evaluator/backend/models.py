from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    total_questions = Column(Integer)
    total_marks = Column(Float)
    max_marks = Column(Float)
    percentage = Column(Float)
    grade = Column(String(5))
    detailed_results = Column(Text)  # JSON string


class DatabaseManager:
    def __init__(self, db_url="sqlite:///evaluations.db"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_evaluation(self, filename, results):
        import json
        session = self.Session()
        try:
            evaluation = Evaluation(
                filename=filename,
                total_questions=results["total_questions"],
                total_marks=results["total_marks"],
                max_marks=results["total_max_marks"],
                percentage=results["percentage"],
                grade=results["grade"],
                detailed_results=json.dumps(results["results"])
            )
            session.add(evaluation)
            session.commit()
            return evaluation.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_all_evaluations(self):
        session = self.Session()
        try:
            evaluations = session.query(Evaluation).order_by(
                Evaluation.upload_date.desc()
            ).all()
            return evaluations
        finally:
            session.close()

    def get_evaluation_by_id(self, eval_id):
        session = self.Session()
        try:
            return session.query(Evaluation).filter_by(id=eval_id).first()
        finally:
            session.close()