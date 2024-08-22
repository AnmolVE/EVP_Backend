from django.db import models
from django.utils.translation import gettext_lazy as _
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager

ROLE_CHOICES = (
    ('Admin', 'Admin'),
    ('User', 'User'),

)

class CustomAccountManager(BaseUserManager):

    def create_superuser(self, email, password, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("role", "Admin")

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Staff must be assigned to is_staff=True.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must be assigned to is_superuser=True.")
        if extra_fields.get("role") != "Admin":
            raise ValueError("Superuser must have role='Admin'.")
        
        return self.create_user(email, password, **extra_fields)
    
    def create_user(self, email, password, **extra_fields):
        if not email:
            raise ValueError(_("You must provide an email address"))
        
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

class NewUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(_("email address"), unique=True)
    role = models.CharField(choices=ROLE_CHOICES, max_length=100,
                            default="User")
    is_staff = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = CustomAccountManager()

    USERNAME_FIELD = "email"

    def __str__(self):
        return self.email

class Company(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    name = models.CharField(max_length=1000, null=True, blank=True)
    headquarters = models.CharField(max_length=1000, null=True, blank=True)
    established_date = models.CharField(max_length=500, null=True, blank=True)
    about_the_company = models.TextField(null=True, blank=True)
    industry = models.TextField(null=True, blank=True)
    company_financials = models.TextField(null=True, blank=True)
    company_history = models.TextField(null=True, blank=True)
    top_3_competitors = models.TextField(null=True, blank=True)
    number_of_employees = models.TextField(null=True, blank=True)
    number_of_geographies = models.TextField(null=True, blank=True)
    linked_info = models.TextField(null=True, blank=True)
    instagram_info = models.TextField(null=True, blank=True)
    tiktok_info = models.TextField(null=True, blank=True)
    facebook_info = models.TextField(null=True, blank=True)
    twitter_info = models.TextField(null=True, blank=True)
    internal_comms_channels = models.TextField(null=True, blank=True)
    exit_interview_feedback = models.TextField(null=True, blank=True)
    employee_feedback_summary = models.TextField(null=True, blank=True)
    engagement_survey_results = models.TextField(null=True, blank=True)
    glassdoor_score = models.TextField(null=True, blank=True)
    online_forums_mentions = models.TextField(null=True, blank=True)
    what_retains_talent = models.TextField(null=True, blank=True)
    what_attracts_talent = models.TextField(null=True, blank=True)
    employee_value_proposition = models.TextField(null=True, blank=True)
    culture_and_values = models.TextField(null=True, blank=True)
    purpose = models.TextField(null=True, blank=True)
    customer_value_proposition = models.TextField(null=True, blank=True)
    vision = models.TextField(null=True, blank=True)
    mission = models.TextField(null=True, blank=True)
    brand_guidelines = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name
    
class TalentDataset(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    area = models.CharField(max_length=500, null=True, blank=True)
    role = models.CharField(max_length=500, null=True, blank=True)
    location = models.CharField(max_length=500, null=True, blank=True)
    seniority = models.CharField(max_length=500, null=True, blank=True)
    key_motivators = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Talent Dataset"
    
class Perception(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    exit_interview_feedback_summary = models.TextField(null=True, blank=True)
    employee_feedback_summary = models.TextField(null=True, blank=True)
    engagement_survey_result_summary = models.TextField(null=True, blank=True)
    glassdoor_score = models.TextField(null=True, blank=True)
    online_forums_summary = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Perception"
    
class Loyalty(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    average_tenure_of_employee = models.TextField(null=True, blank=True)
    net_promoter_score = models.TextField(null=True, blank=True)
    number_of_early_exits = models.TextField(null=True, blank=True)
    number_of_re_hires = models.TextField(null=True, blank=True)
    what_retains_talent = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Loyalty"
    
class Advocacy(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    number_of_employees = models.TextField(null=True, blank=True)
    number_of_referrals = models.TextField(null=True, blank=True)
    number_of_referrals_to_hires = models.TextField(null=True, blank=True)
    esat_recommendability_score = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Advocacy"
    
class Attraction(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    number_of_jobs_posted = models.TextField(null=True, blank=True)
    average_number_of_job_post_clicks = models.TextField(null=True, blank=True)
    number_of_direct_hires = models.TextField(null=True, blank=True)
    average_time_to_fill = models.TextField(null=True, blank=True)
    number_of_offers_made = models.TextField(null=True, blank=True)
    number_of_offers_accepted = models.TextField(null=True, blank=True)
    number_of_direct_applicants = models.TextField(null=True, blank=True)
    number_of_hires = models.TextField(null=True, blank=True)
    what_attracts_talent = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Attraction"
    
class Influence(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    number_of_career_page_subscribers = models.TextField(null=True, blank=True)
    number_of_views = models.TextField(null=True, blank=True)
    engagement = models.TextField(null=True, blank=True)
    number_of_media_mentions = models.TextField(null=True, blank=True)
    number_of_awards = models.TextField(null=True, blank=True)
    summary_of_awards_or_recognition = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Influence"
    
class Brand(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    employee_value_proposition = models.TextField(null=True, blank=True)
    culture_and_values = models.TextField(null=True, blank=True)
    purpose = models.TextField(null=True, blank=True)
    customer_value_proposition = models.TextField(null=True, blank=True)
    vision = models.TextField(null=True, blank=True)
    mission = models.TextField(null=True, blank=True)
    internal_comms_samples = models.TextField(null=True, blank=True)
    external_comms_samples = models.TextField(null=True, blank=True)
    brand_guidelines = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Brand"

class AttributesOfGreatPlace(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    culture = models.TextField(null=True, blank=True)
    purpose_and_values = models.TextField(null=True, blank=True)
    benefits_perks = models.TextField(null=True, blank=True)
    career_development = models.TextField(null=True, blank=True)
    office_and_facilities = models.TextField(null=True, blank=True)
    leadership_and_management = models.TextField(null=True, blank=True)
    rewards_and_recognition= models.TextField(null=True, blank=True)
    teamwork_and_collaboration = models.TextField(null=True, blank=True)
    brand_and_reputation = models.TextField(null=True, blank=True)
    work_life_balance = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Attributes of Great Place"
    
class KeyThemes(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    top_key_themes = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Key Themes"
    
class AudienceWiseMessaging(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    existing_employees = models.TextField(null=True, blank=True)
    alumni = models.TextField(null=True, blank=True)
    targeted_talent = models.TextField(null=True, blank=True)
    leadership = models.TextField(null=True, blank=True)
    recruiters = models.TextField(null=True, blank=True)
    clients = models.TextField(null=True, blank=True)
    offer_drops = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Audience-Wise Messaging"

class SwotAnalysis(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    what_is_working_well_for_the_organization = models.TextField(null=True, blank=True)
    what_is_not_working_well_for_the_organization = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Swot Analysis"
    
class Alignment(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    what_we_want_to_be_known_for = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Alignment"

class MessagingHierarchyTabs(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    tab_name = models.TextField(null=True, blank=True)
    tabs_data = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.tab_name}"
    
class MessagingHierarchyData(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    main_theme = models.TextField(null=True, blank=True)
    pillar_1 = models.TextField(null=True, blank=True)
    pillar_2 = models.TextField(null=True, blank=True)
    pillar_3 = models.TextField(null=True, blank=True)
    tagline = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - MessagingHierarchyData"
    
class CreativeDirection(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    creative_direction_data = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Creative Direction"
    
class EVPDefinition(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    theme = models.TextField(null=True, blank=True)
    what_it_means = models.TextField(null=True, blank=True)
    what_it_does_not_mean = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.theme}"
    
class EVPPromise(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    theme = models.TextField(null=True, blank=True)
    what_employees_can_expect = models.TextField(null=True, blank=True)
    what_is_expected_of_employees = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.theme}"

class EVPAudit(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    theme = models.TextField(null=True, blank=True)
    what_makes_this_credible = models.TextField(null=True, blank=True)
    where_do_we_need_to_stretch = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.theme}"
    
class EVPEmbedmentStage(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    stage_name = models.CharField(max_length=255)

    def __str__(self):
        return self.stage_name
    
class EVPEmbedmentTouchpoint(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    stage = models.ForeignKey(EVPEmbedmentStage, related_name='touchpoints', on_delete=models.CASCADE)
    touchpoint_name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.stage.stage_name} - {self.touchpoint_name}"
    
class EVPEmbedmentMessage(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    touchpoint = models.OneToOneField(EVPEmbedmentTouchpoint, related_name='message', on_delete=models.CASCADE)
    message = models.TextField()

    def __str__(self):
        return f"{self.touchpoint.stage.stage_name} - {self.touchpoint.touchpoint_name}"
    
class EVPHandbook(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    handbook_data = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - EVPHandbook"
    
class EVPStatementAndPillars(models.Model):
    user = models.ForeignKey(NewUser, default=None, on_delete=models.CASCADE)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    evp_statement_document = models.FileField(upload_to="evp_statement_documents/", null=True, blank=True)
    evp_statement_thumbnail = models.ImageField(upload_to="evp_statement_thumbnails/", null=True, blank=True)
    evp_statement_text = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.company.name} - Statement and Pillars"
    