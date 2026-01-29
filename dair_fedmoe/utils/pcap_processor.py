"""
PCAP processing utilities for DAIR-FedMoE.
"""

import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, Raw, wrpcap
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)

class PCAPProcessor:
    """Processor for PCAP files to extract features for encrypted traffic classification."""
    
    def __init__(
        self,
        max_packets_per_flow: int = 10,
        max_payload_size: int = 1500,  # Ethernet MTU
        flow_timeout: int = 600,  # 10 minutes in seconds
        include_headers: bool = True,
        include_stats: bool = True,
        editcap_path: str = "editcap",
        splitcap_path: str = "SplitCap.exe"
    ):
        self.max_packets_per_flow = max_packets_per_flow
        self.max_payload_size = max_payload_size
        self.flow_timeout = flow_timeout
        self.include_headers = include_headers
        self.include_stats = include_stats
        self.editcap_path = editcap_path
        self.splitcap_path = splitcap_path
        
    def _convert_pcapng_to_pcap(self, pcapng_path: str) -> str:
        """Convert pcapng file to pcap format using EditCap."""
        try:
            # Create temporary file for pcap
            temp_dir = tempfile.mkdtemp()
            pcap_path = os.path.join(temp_dir, os.path.basename(pcapng_path).replace('.pcapng', '.pcap'))
            
            # Run EditCap command
            cmd = [self.editcap_path, '-F', 'pcap', pcapng_path, pcap_path]
            subprocess.run(cmd, check=True, capture_output=True)
            
            return pcap_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting {pcapng_path} to pcap: {e.stderr.decode()}")
            raise
            
    def _split_into_sessions(self, pcap_path: str) -> str:
        """Split pcap file into bidirectional traffic sessions using SplitCap."""
        try:
            # Create output directory
            output_dir = os.path.join(os.path.dirname(pcap_path), 'sessions')
            os.makedirs(output_dir, exist_ok=True)
            
            # Run SplitCap command
            cmd = [
                self.splitcap_path,
                '-p', str(self.flow_timeout),  # Flow timeout
                '-o', output_dir,  # Output directory
                '-s', 'flow',  # Split by flow
                pcap_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            return output_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Error splitting {pcap_path} into sessions: {e.stderr.decode()}")
            raise
            
    def _pad_udp_headers(self, packet) -> None:
        """Pad UDP headers to match TCP header size."""
        if UDP in packet:
            # Calculate padding needed to match TCP header size (20 bytes)
            padding_needed = 20 - len(packet[UDP])
            if padding_needed > 0:
                # Add padding bytes
                packet[UDP].add_payload(b'\x00' * padding_needed)
                
    def _normalize_packet_data(self, packet) -> None:
        """Convert packet data to raw bytes and normalize to 0.0-1.0 range."""
        if packet.haslayer(Raw):
            # Convert payload to bytes
            payload = bytes(packet[Raw].load)
            
            # Pad to max_payload_size
            if len(payload) < self.max_payload_size:
                payload += b'\x00' * (self.max_payload_size - len(payload))
            else:
                payload = payload[:self.max_payload_size]
                
            # Normalize to 0.0-1.0 range
            normalized = np.frombuffer(payload, dtype=np.uint8).astype(np.float32) / 255.0
            
            # Update packet payload
            packet[Raw].load = normalized.tobytes()
            
    def _preprocess_packet(self, packet) -> None:
        """Apply all preprocessing steps to a packet."""
        self._pad_udp_headers(packet)
        self._normalize_packet_data(packet)
        
    def _extract_flow_key(self, packet) -> Optional[Tuple]:
        """Extract flow key (src_ip, dst_ip, src_port, dst_port, protocol) from packet."""
        if not (IP in packet and (TCP in packet or UDP in packet)):
            return None
            
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto
        
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        else:  # UDP
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            
        # Sort IPs and ports to ensure consistent flow direction
        if src_ip > dst_ip or (src_ip == dst_ip and src_port > dst_port):
            src_ip, dst_ip = dst_ip, src_ip
            src_port, dst_port = dst_port, src_port
            
        return (src_ip, dst_ip, src_port, dst_port, protocol)
        
    def _extract_packet_features(self, packet) -> Dict:
        """Extract features from a single packet."""
        features = {}
        
        # Basic packet features
        features['length'] = len(packet)
        features['time'] = packet.time
        
        if IP in packet:
            features['ip_version'] = packet[IP].version
            features['ip_ttl'] = packet[IP].ttl
            features['ip_tos'] = packet[IP].tos
            
        if TCP in packet:
            features['tcp_flags'] = packet[TCP].flags
            features['tcp_window'] = packet[TCP].window
            features['tcp_options'] = len(packet[TCP].options)
        elif UDP in packet:
            features['udp_length'] = packet[UDP].len
            
        # Extract payload (if any)
        if packet.haslayer(Raw):
            payload = packet[Raw].load
            features['payload'] = payload
        else:
            features['payload'] = np.zeros(self.max_payload_size, dtype=np.float32)
            
        return features
        
    def _extract_flow_features(self, packets: List) -> Dict:
        """Extract features from a flow of packets."""
        if not packets:
            return {}
            
        # Sort packets by time
        packets.sort(key=lambda x: x.time)
        
        # Extract packet-level features
        packet_features = [self._extract_packet_features(p) for p in packets]
        
        # Flow-level features
        flow_features = {
            'num_packets': len(packets),
            'duration': packets[-1].time - packets[0].time,
            'total_bytes': sum(p['length'] for p in packet_features),
            'avg_packet_size': np.mean([p['length'] for p in packet_features]),
            'std_packet_size': np.std([p['length'] for p in packet_features]),
        }
        
        # Add header features if requested
        if self.include_headers:
            header_features = {}
            for i, p in enumerate(packet_features[:self.max_packets_per_flow]):
                for k, v in p.items():
                    if k != 'payload':
                        header_features[f'p{i+1}_{k}'] = v
            flow_features.update(header_features)
            
        # Add statistical features if requested
        if self.include_stats:
            stats_features = {
                'packet_size_entropy': self._calculate_entropy([p['length'] for p in packet_features]),
                'inter_arrival_times': self._calculate_inter_arrival_times(packets),
                'protocol_ratio': self._calculate_protocol_ratio(packets),
            }
            flow_features.update(stats_features)
            
        # Add payload features
        payload_features = self._extract_payload_features(packet_features)
        flow_features.update(payload_features)
        
        return flow_features
        
    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of a list of values."""
        if not values:
            return 0.0
        hist, _ = np.histogram(values, bins=50, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
        
    def _calculate_inter_arrival_times(self, packets: List) -> Dict:
        """Calculate inter-arrival time statistics."""
        if len(packets) < 2:
            return {'mean_iat': 0, 'std_iat': 0}
            
        iats = [packets[i].time - packets[i-1].time for i in range(1, len(packets))]
        return {
            'mean_iat': np.mean(iats),
            'std_iat': np.std(iats)
        }
        
    def _calculate_protocol_ratio(self, packets: List) -> Dict:
        """Calculate protocol ratios in the flow."""
        total = len(packets)
        if total == 0:
            return {'tcp_ratio': 0, 'udp_ratio': 0}
            
        tcp_count = sum(1 for p in packets if TCP in p)
        udp_count = sum(1 for p in packets if UDP in p)
        
        return {
            'tcp_ratio': tcp_count / total,
            'udp_ratio': udp_count / total
        }
        
    def _extract_payload_features(self, packet_features: List[Dict]) -> Dict:
        """Extract features from packet payloads."""
        payload_features = {}
        
        # Take first N packets
        for i, p in enumerate(packet_features[:self.max_packets_per_flow]):
            payload = p['payload']
            if isinstance(payload, bytes):
                # Convert bytes to normalized array
                payload_array = np.frombuffer(payload, dtype=np.uint8).astype(np.float32) / 255.0
            else:
                payload_array = payload  # Already normalized
                
            # Basic payload statistics
            payload_features[f'p{i+1}_payload_length'] = len(payload_array)
            payload_features[f'p{i+1}_payload_entropy'] = self._calculate_entropy(payload_array)
            payload_features[f'p{i+1}_payload'] = payload_array
                
        return payload_features
        
    def process_pcap_file(self, pcap_path: str) -> List[Dict]:
        """Process a single PCAP file and extract flow features."""
        try:
            # Convert pcapng to pcap if needed
            if pcap_path.endswith('.pcapng'):
                pcap_path = self._convert_pcapng_to_pcap(pcap_path)
                
            # Split into sessions
            sessions_dir = self._split_into_sessions(pcap_path)
            
            # Process each session
            all_flows = []
            for session_file in os.listdir(sessions_dir):
                if session_file.endswith('.pcap'):
                    session_path = os.path.join(sessions_dir, session_file)
                    
                    # Read session packets
                    packets = rdpcap(session_path)
                    
                    # Preprocess packets
                    for packet in packets:
                        self._preprocess_packet(packet)
                        
                    # Extract features
                    features = self._extract_flow_features(packets)
                    if features:
                        features['flow_key'] = session_file
                        all_flows.append(features)
                        
            # Clean up temporary files
            shutil.rmtree(os.path.dirname(pcap_path))
            shutil.rmtree(sessions_dir)
            
            return all_flows
            
        except Exception as e:
            logger.error(f"Error processing {pcap_path}: {str(e)}")
            return []
            
    def process_directory(
        self,
        pcap_dir: str,
        num_workers: int = 4
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Process all PCAP files in a directory."""
        pcap_files = []
        for root, _, files in os.walk(pcap_dir):
            for file in files:
                if file.endswith(('.pcap', '.pcapng')):
                    pcap_files.append(os.path.join(root, file))
                    
        # Process files in parallel
        all_flows = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_pcap_file, f) for f in pcap_files]
            for future in tqdm(futures, desc="Processing PCAP files"):
                flows = future.result()
                all_flows.extend(flows)
                
        # Convert to DataFrame
        df = pd.DataFrame(all_flows)
        
        # Extract labels from filenames
        labels = [os.path.basename(f).split('_')[0] for f in pcap_files]
        
        return df, labels 