from django.contrib.auth import authenticate

from rest_framework import serializers
from .models import (
    NewUser,
    Company,
    Perception,
    Loyalty,
    Advocacy,
    Attraction,
    Influence,
    Brand,
    AttributesOfGreatPlace,
    KeyThemes,
    AudienceWiseMessaging,
    SwotAnalysis,
    Alignment,
    MessagingHierarchyTabs,
    MessagingHierarchyData,
    EVPPromise,
    EVPAudit,
)

class NewUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewUser
        fields = ["id", "email"]

class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')
        
        if email and password:
            user = authenticate(email=email, password=password)
            if user:
                if not user.is_active:
                    raise serializers.ValidationError("User account is disabled.")
                data['user'] = user
            else:
                raise serializers.ValidationError("Unable to log in with provided credentials.")
        else:
            raise serializers.ValidationError("Must include 'email' and 'password'.")
        
        return data

class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = '__all__'

class PerceptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Perception
        fields = '__all__'

class LoyaltySerializer(serializers.ModelSerializer):
    class Meta:
        model = Loyalty
        fields = '__all__'

class AdvocacySerializer(serializers.ModelSerializer):
    class Meta:
        model = Advocacy
        fields = '__all__'

class AttractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attraction
        fields = '__all__'

class InfluenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Influence
        fields = '__all__'

class BrandSerializer(serializers.ModelSerializer):
    class Meta:
        model = Brand
        fields = '__all__'

class MessagingHierarchyTabsSerializer(serializers.ModelSerializer):
    class Meta:
        model = MessagingHierarchyTabs
        fields = "__all__"

class MessagingHierarchyDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MessagingHierarchyData
        fields = "__all__"

class AttributesOfGreatPlaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = AttributesOfGreatPlace
        fields = "__all__"

class KeyThemesSerializer(serializers.ModelSerializer):
    class Meta:
        model = KeyThemes
        fields = "__all__"

class AudienceWiseMessagingSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudienceWiseMessaging
        fields = "__all__"

class SwotAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = SwotAnalysis
        fields = "__all__"

class AlignmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alignment
        fields = "__all__"

class EVPPromiseSerializer(serializers.ModelSerializer):
    class Meta:
        model = EVPPromise
        fields = "__all__"

class EVPAuditSerializer(serializers.ModelSerializer):
    class Meta:
        model = EVPAudit
        fields = "__all__"
