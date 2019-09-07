from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField, SelectField
from wtforms.validators import DataRequired

class ReviewForm(FlaskForm):
    reviewText = TextAreaField('Review Text', validators=[DataRequired()], render_kw={"rows": 8, "cols": 6})
    resultPresentation = SelectField('Product Result Type',
    choices =[('tr', 'Top Result'), ('tt', 'Top Three')])
    submit = SubmitField('Classify')