from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import *

class NewUserAdmin(UserAdmin):
    model = NewUser
    list_display = ['email', 'is_staff', 'is_active', 'created_at', 'updated_at']
    ordering = ['email']
    search_fields = ('email',)
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ()}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser')}),
        ('Important dates', {'fields': ('last_login', 'created_at', 'updated_at')}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'is_active', 'is_staff', 'is_superuser'),
        }),
    )

    readonly_fields = ('created_at', 'updated_at')

class CompanyAdmin(admin.ModelAdmin):
    list_display = ["id", "user", 'name', 'headquarters', 'established_date', 'about_the_company', 'industry', 'company_financials', 'company_history', 'top_3_competitors', 'number_of_employees', 'number_of_geographies', 'linked_info', 'instagram_info', 'tiktok_info', 'facebook_info', 'twitter_info', 'internal_comms_channels']
    list_filter = ['industry']

class PerceptionAdmin(admin.ModelAdmin):
    list_display = ["id", "user", 'company', 'exit_interview_feedback_summary', 'employee_feedback_summary', 'engagement_survey_result_summary', 'glassdoor_score', 'online_forums_summary']
    list_filter = ['company']

class LoyaltyAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "average_tenure_of_employee", "net_promoter_score", "number_of_early_exits", "number_of_re_hires", "what_retains_talent"]
    list_filter = ["company"]

class AdvocacyAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "number_of_employees", "number_of_referrals", "number_of_referrals_to_hires", "esat_recommendability_score"]
    list_filter = ["company"]

class AttractionAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "number_of_jobs_posted", "average_number_of_job_post_clicks", "number_of_direct_hires", "average_time_to_fill", "number_of_offers_made", "number_of_offers_accepted", "number_of_direct_applicants", "number_of_hires", "what_attracts_talent"]
    list_filter = ["company"]

class InfluenceAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "number_of_career_page_subscribers", "number_of_views", "engagement", "number_of_media_mentions", "number_of_awards", "summary_of_awards_or_recognition"]
    list_filter = ["company"]

class BrandAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "employee_value_proposition", "culture_and_values", "purpose", "customer_value_proposition", "vision", "mission", "internal_comms_samples", "external_comms_samples", "brand_guidelines"]
    list_filter = ["company"]

class AttributesOfGreatPlaceAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "culture", "purpose_and_values", "benefits_perks", "career_development", "office_and_facilities", "office_and_facilities", "leadership_and_management", "rewards_and_recognition", "teamwork_and_collaboration", "brand_and_reputation", "work_life_balance"]
    list_filter = ["company"]

class KeyThemesAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "top_key_themes"]
    list_filter = ["company"]

class AudienceWiseMessagingAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "existing_employees", "alumni", "targeted_talent", "leadership", "recruiters", "clients", "offer_drops"]
    list_filter = ["company"]

class SwotAnalysisAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "what_is_working_well_for_the_organization", "what_is_not_working_well_for_the_organization"]
    list_filter = ["company"]

class AlignmentAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "what_we_want_to_be_known_for"]
    list_filter = ["company"]

class MessagingHierarchyTabsAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "tab_name", "tabs_data"]
    list_filter = ["company"]

class MessagingHierarchyDataAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "main_theme", "pillar_1", "pillar_2", "pillar_3", "tagline"]
    list_filter = ["company"]

class EVPPromiseAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "theme", "what_makes_this_theme_distinctive", "what_employees_can_expect", "what_is_expected_of_employees"]
    list_filter = ["company"]

class EVPAuditAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "theme", "what_makes_this_credible", "where_do_we_need_to_stretch"]
    list_filter = ["company"]

class EVPEmbedmentStageAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "stage_name"]
    list_filter = ["company"]

class EVPEmbedmentTouchpointAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "stage", "touchpoint_name"]
    list_filter = ["company"]

class EVPEmbedmentMessageAdmin(admin.ModelAdmin):
    list_display = ["id", "user", "company", "touchpoint", "message"]
    list_filter = ["company"]

admin.site.register(NewUser, NewUserAdmin)
admin.site.register(Company, CompanyAdmin)
admin.site.register(Perception, PerceptionAdmin)
admin.site.register(Loyalty, LoyaltyAdmin)
admin.site.register(Advocacy, AdvocacyAdmin)
admin.site.register(Attraction, AttractionAdmin)
admin.site.register(Influence, InfluenceAdmin)
admin.site.register(Brand, BrandAdmin)
admin.site.register(AttributesOfGreatPlace, AttributesOfGreatPlaceAdmin)
admin.site.register(KeyThemes, KeyThemesAdmin)
admin.site.register(AudienceWiseMessaging, AudienceWiseMessagingAdmin)
admin.site.register(SwotAnalysis, SwotAnalysisAdmin)
admin.site.register(Alignment, AlignmentAdmin)
admin.site.register(MessagingHierarchyTabs, MessagingHierarchyTabsAdmin)
admin.site.register(MessagingHierarchyData, MessagingHierarchyDataAdmin)
admin.site.register(EVPPromise, EVPPromiseAdmin)
admin.site.register(EVPAudit, EVPAuditAdmin)
admin.site.register(EVPEmbedmentStage, EVPEmbedmentStageAdmin)
admin.site.register(EVPEmbedmentTouchpoint, EVPEmbedmentTouchpointAdmin)
admin.site.register(EVPEmbedmentMessage, EVPEmbedmentMessageAdmin)