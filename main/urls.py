from django.urls import path
from .views import *

from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path("testing/", Testing.as_view(), name="testing"),
    path("test-bing/", TestBingAPIView.as_view(), name="test-bing"),

    path("token/", MyTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),

    path("login/", LoginAPIView.as_view(), name="login"),
    path("master-vector-database/", MasterVectorDatabaseAPIView.as_view(), name="master-vector-database"),
    path("home-page/", homePageAPIView.as_view(), name="home-page"),
    path("talent-insights-home/", TalentInsightsHomeAPIView.as_view(), name="talent-insights-home"),
    path("industry-trends-home/", IndustryTrendsHomeAPIView.as_view(), name="industry-trends-home"),
    path("chatbot/", ChatBotAPIView.as_view(), name="search"),
    
    path("search/", SearchWebsiteView.as_view(), name="search"),
    path("design/", DesignAPIView.as_view(), name="design"),
    path("themes-regenerate/", Top4ThemesRegenerateAPIView.as_view(), name="themes-regenerate"),
    path("send-mail/", SendMailAPIView.as_view(), name="send-mail"),
    path("transcript/<str:company_name>/", TranscriptAPIView.as_view(), name="transcript"),
    path("design-principles/", DesignPrinciplesAPIView.as_view(), name="design-principles"),
    path("design-principles/<str:company_name>/", DesignPrinciplesSpecificAPIView.as_view(), name="design-principles-specific"),
    path("talent-dataset/", TalentDatasetAPIView.as_view(), name="talent-dataset"),
    path("companies/<str:company_name>/", CompanySpecificAPIView.as_view(), name='company-specific'),
    path("perception/<str:company_name>/", PerceptionSpecificAPIView.as_view(), name='perception-specific'),
    path("loyalty/<str:company_name>/", LoyaltySpecificAPIView.as_view(), name='loyalty-specific'),
    path("advocacy/<str:company_name>/", AdvocacySpecificAPIView.as_view(), name='advocacy-specific'),
    path("attraction/<str:company_name>/", AttractionSpecificAPIView.as_view(), name='attraction-specific'),
    path("influence/<str:company_name>/", InfluenceSpecificAPIView.as_view(), name='influence-specific'),
    path("brand/<str:company_name>/", BrandSpecificAPIView.as_view(), name='brand-specific'),
    path("attributes-of-great-workplace/", AttributesOfGreatPlaceAPIView.as_view(), name='attributes-of-great-workplace'),
    path("attributes-of-great-workplace/<str:company_name>/", AttributesOfGreatPlaceSpecificAPIView.as_view(), name='attributes-of-great-workplace-specific'),
    path("key-themes/", KeyThemesAPIView.as_view(), name='key-themes'),
    path("key-themes/<str:company_name>/", KeyThemesSpecificAPIView.as_view(), name='key-themes-specific'),
    path("audience-wise-messaging/", AudienceWiseMessagingAPIView.as_view(), name='audience-wise-messaging'),
    path("audience-wise-messaging/<str:company_name>/", AudienceWiseMessagingSpecificAPIView.as_view(), name='audience-wise-messaging-specific'),
    path("talent-insights/", TalentInsightsAPIView.as_view(), name="talent-insights"),
    path("swot-analysis/", SwotAnalysisAPIView.as_view(), name='swot-analysis'),
    path("swot-analysis/<str:company_name>/", SwotAnalysisSpecificAPIView.as_view(), name='swot-analysis-specific'),
    path("alignment/", AlignmentAPIView.as_view(), name='alignment'),
    path("alignment/<str:company_name>/", AlignmentSpecificAPIView.as_view(), name='alignment-specific'),
    path("messaging-hierarchy/<str:company_name>/", MessagingHierarchySpecificAPIView.as_view(), name='messaging-hierarchy'),
    path("tagline/", TaglineAPIView.as_view(), name='tagline'),
    path("creative-direction/", CreativeDirectionAPIView.as_view(), name="creative-direction"),
    path("evp-definition/", EVPDefinitionAPIView.as_view(), name="evp-definition"),
    path("evp-promise/", EVPPromiseAPIView.as_view(), name="evp-promise"),
    path("evp-audit/", EVPAuditAPIView.as_view(), name="evp-audit"),
    path("evp-embedment/", EVPEmbedmentAPIView.as_view(), name="evp-embedment"),
    path("evp-narrative/", EVPNarrativeAPIView.as_view(), name="evp-narrative"),
    path("evp-handbook/", EVPHandBookAPIView.as_view(), name="evp-handbook"),
    path("evp-execution-plan/<str:company_name>/", EVPExecutionPlanSpecificAPIView.as_view(), name="evp-execution-plan"),
    path("evp-statement-and-pillars/<str:company_name>/", EVPStatementAndPillarsSpecificAPIView.as_view(), name="evp-statement-and-pillars"),


    # *******************Module 2 - Internal Communication***********************


    path("icicsi/", ICICSIAPIView.as_view(), name="icicsi"),
    path("icicsi/<str:company_name>/", ICICSISpecificAPIView.as_view(), name="icicsi-specific")
]
