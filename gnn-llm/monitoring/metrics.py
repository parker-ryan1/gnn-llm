import time
import psutil
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import thread
from coll

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float

@dataclass
ccs:
tetime
    stage:
    duration_sat
    records_procesed: int
    success: bool
    error_message: str = ""

classctor:
    def __ = 1000):
        self.max_history y
)
        se
        self.logg__)
        self.monitoringalse
        self.mne
     
    def start_ 30):
        """Sta
        if self.monitoring_active:
return
        
        self.moniue
        self.mon
            ta
            args=(interval_secon),
    ue
        )
        self.monitor_thread.start()
erval")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
           
        self.logger.info("Stopped system monitoring"
    
    def _monitor_system(self, intert):
        """Monitor system metrics cont
        while self.monitoring_active:
           try:
                metrics = self.collect_system_metrics()
             rics)
                
                # sage
        80:
                    self.logger.warni%")
                if metrics.memory_percent80:
                    self.logger.warning(")
                
                time.sl)
         
              }")
          
    
    def collect_system_metricsmMetrics:
        """C"
        memory = psutil.virtual_memory()
        
        mMetrics(
            timestamp=datetime.now(),
    
            memory_percent=memory.percent,
           4**3),
            memory_total_gb=memory.to)
        )
    
    def record_pipeline_stageint, 
                
        """Record pipecs"""
        metrics = PipelineMetrics(
            timestamp=datetime.now(),
            stage=stage,
            dtion,
          records,
            success=success,
            errosage=error
        )
        self.pipeline_metrics.appe
        
        log_msg = f"Pipeline stage '{stage}'ds"
    ess:
            self.logger.info(log_msg)
        el
            self.logger.error(f"{log_
    
    def 
        """Generate a perform"
        if not self.system_metricsics:
            return "No metrics available"
        
        recent_]
        pipeline_runs = list(self.pipeline_metrics)
        
    )
        avg_memory = sum(m.memory_percent for m m)
        
    s]
        success_rate = len(sse 0
        
        report = f"""
=== PERFORMAN=
System Metrics:
- Average CPU: {avg_cpu:
- Average Memory: {avg_memory:.1f}%

Pipeline Metrics:
- Total Runs: {len(pipeline_}
- Success %}
- Records)}
        """
        
        return report.strip()

# Global metrics collector
metrics_colleor()sCollectic Metr =ctor