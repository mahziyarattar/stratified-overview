from stratified_overview.cleaning import RepeatedParagraphFilter
from stratified_overview.models import Document


def test_drops_paragraphs_repeated_across_documents():
    boilerplate = "Sign up to our free newsletter"
    docs = [
        Document(id=str(i), text=f"Story {i} begins here.\n{boilerplate}\n  sign up to our FREE newsletter ")
        for i in range(3)
    ]
    outcome = RepeatedParagraphFilter(min_documents=3).clean(docs)
    assert outcome.paragraphs_removed == 6
    assert all(d.text == f"Story {d.id} begins here." for d in outcome.documents)


def test_keeps_paragraphs_below_threshold():
    docs = [Document(id="a", text="Shared line\nA"), Document(id="b", text="Shared line\nB")]
    outcome = RepeatedParagraphFilter(min_documents=3).clean(docs)
    assert outcome.paragraphs_removed == 0
    assert outcome.documents == docs
