from django.db import models


class Flight(models.Model):
    flight_number = models.CharField(max_length=10, unique=True)
    airline = models.CharField(max_length=100)
    airplane = models.CharField(max_length=100)
    departure_city = models.CharField(max_length=100)
    arrival_city = models.CharField(max_length=100)
    departure_time = models.TimeField()
    arrival_time = models.TimeField()
    date = models.DateField()

    def __str__(self):
        return f"{self.flight_number} ({self.airline}) - {self.date}"


class FlightClass(models.Model):
    CLASS_CHOICES = [
        ('ECONOMY', 'Economy'),
        ('BUSINESS', 'Business'),
        ('FIRST', 'First'),
    ]

    flight = models.ForeignKey(Flight, on_delete=models.CASCADE, related_name='classes')
    class_type = models.CharField(max_length=10, choices=CLASS_CHOICES)
    sold_count = models.PositiveIntegerField(default=0)
    price = models.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        unique_together = ('flight', 'class_type')

    def __str__(self):
        return f"{self.flight.flight_number} - {self.class_type}"

class Test(models.Model):
    flight = models.ForeignKey(Flight, on_delete=models.CASCADE, related_name='test')


class Test2(models.Model):
    flight = models.ForeignKey(Flight, on_delete=models.CASCADE, related_name='test2')
