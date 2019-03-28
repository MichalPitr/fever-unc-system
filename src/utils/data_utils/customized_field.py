from allennlp.data.fields import MetadataField


class IdField(MetadataField):
    """
    This is only a customized Id field that override meta data __str__ to show ids.
    """
    def __str__(self) -> str:
        return f"IdField with id: {self.metadata}."
