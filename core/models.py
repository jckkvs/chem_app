from django.db import models


class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=1024)
    smiles_col = models.CharField(max_length=255, default='SMILES')
    target_col = models.CharField(max_length=255, default='target')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Experiment(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='PENDING')
    config = models.JSONField(default=dict)  # Features used, model params
    metrics = models.JSONField(default=dict, blank=True, null=True)
    artifacts_path = models.CharField(max_length=1024, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.status})"


class InverseDesignJob(models.Model):
    """逆解析ジョブ"""
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('RUNNING', 'Running'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    target_property = models.CharField(max_length=255)
    target_value = models.FloatField()
    direction = models.CharField(max_length=20)  # maximize, minimize, target
    method = models.CharField(max_length=50)  # exhaustive, random, bayesian
    n_iterations = models.IntegerField(default=100)
    constraints = models.JSONField(default=dict)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='PENDING')
    results = models.JSONField(default=dict, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"InverseDesign {self.id} ({self.status})"
