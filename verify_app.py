"""Exercise all five Streamlit forms and prediction buttons without a browser."""
from streamlit.testing.v1 import AppTest

PAGES = ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction',
         'Lung Cancer Prediction', 'Hypo-Thyroid Prediction']
app = AppTest.from_file('app.py', default_timeout=30).run()
assert not app.exception, app.exception
for page in PAGES:
    app.selectbox[0].select(page).run()
    assert not app.exception, (page, app.exception)
    app.button[1].click().run()
    assert not app.exception, (page, app.exception)
    assert any('Complete every field' in error.value for error in app.error), page
    app.button[0].click().run()
    assert not app.exception, (page, app.exception)
    assert all(widget.value is not None for widget in app.number_input), page
    app.button[1].click().run()
    assert not app.exception, (page, app.exception)
    assert len(app.success) == 1, (page, app.error)
    print(page, app.success[0].value, flush=True)
print('All five forms, missing-input validation, and predictions passed.')
