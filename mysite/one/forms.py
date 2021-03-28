from django.forms import ModelForm
class PostForm(ModelForm):
    class Meta:
        model = Post
        fields = ['text']
        widgets = {
            'text': forms.TextInput(attrs={
                'id': 'text', 
                'required': True, 
                'placeholder': 'Article text...'
            }),
        }