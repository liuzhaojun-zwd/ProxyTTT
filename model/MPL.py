import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MPLModule(nn.Module):
    """
    Modified to handle non-contiguous labels with a dynamic member bank.
    Enhanced with local reconstruction loss for fine-grained feature learning.
    Added margin-based learning to increase training difficulty and improve feature discrimination.
    """
    def __init__(self, feature_dim, num_proxies=5, temp=0.1, proxy_update_momentum=0.9, 
                 mask_ratio=0.15, recon_weight=1.0, use_frl=True, 
                 margin=0.3, margin_type='cos'):
        """
        Initialize the JPL Module for single modality.
        
        Args:
            feature_dim (int): Dimension of the feature representations
            num_proxies (int): Number of proxies per class (Np in the paper)
            temp (float): Temperature parameter for similarity scaling
            proxy_update_momentum (float): Momentum factor for proxy updates
            mask_ratio (float): Ratio of feature dimensions to mask for reconstruction
            recon_weight (float): Weight of reconstruction loss in total loss
            use_local_recon (bool): Whether to use local reconstruction loss
            margin (float): Margin value for margin-based learning
            margin_type (str): Type of margin to use ('cos' for cosine margin, 'arc' for arcface)
        """
        super(MPLModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_proxies = num_proxies
        self.temp = temp
        self.proxy_update_momentum = proxy_update_momentum
        self.mask_ratio = mask_ratio
        self.recon_weight = recon_weight
        self.use_local_recon = use_frl
        
        # Margin parameters
        self.margin = margin
        self.margin_type = margin_type
        
        # Dynamic member bank to store class-specific data
        
        self.member_bank = {}
        
        # Keep track of registered class IDs
        self.register_buffer('registered_classes', torch.tensor([], dtype=torch.long))
        
        if self.use_local_recon:
             # Reconstruction projectors
            self.encoder = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        
    def _create_class_entry(self, class_id, device):
        """
        Create a new class entry in the member bank
        
        Args:
            class_id (int): Original class ID
            device (torch.device): Device to create tensors on
        """
        # Create proxies for this class
        proxies = F.normalize(
            torch.randn(self.num_proxies, self.feature_dim, device=device),
            p=2, 
            dim=-1
        )
        
        # Only keep necessary components in member bank
        self.member_bank[class_id] = {
            'proxies': nn.Parameter(proxies),
            'running_features': torch.zeros(self.num_proxies, self.feature_dim, device=device),
            'feature_counts': torch.zeros(self.num_proxies, device=device)
        }
        
        # Add the parameter to the module
        self.register_parameter(f'proxies_{class_id}', self.member_bank[class_id]['proxies'])
        
        # Update registered classes
        self.registered_classes = torch.cat([
            self.registered_classes,
            torch.tensor([class_id], device=device, dtype=torch.long)
        ])
        
    def register_new_labels(self, labels, device):
        """
        Register new label IDs and create entries in the member bank
        
        Args:
            labels (tensor): Original label IDs [batch_size]
            device (torch.device): Device to create tensors on
        """
        # Get unique label IDs in the batch
        unique_labels = labels.unique().tolist()
        
        # Check for new labels and register them
        for label_id in unique_labels:
            if label_id not in self.member_bank:
                self._create_class_entry(label_id, device)
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle dynamically created proxy parameters
        """
        # First, identify proxy parameters in state_dict
        proxy_params = {}
        other_params = {}
        
        for key, value in state_dict.items():
            if key.startswith('proxies_'):
                proxy_params[key] = value
            else:
                other_params[key] = value
        
        # Load non-proxy parameters first
        super().load_state_dict(other_params, strict=False)
        
        # Extract class IDs from proxy parameter names and create entries
        class_ids_to_register = []
        for param_name, param_value in proxy_params.items():
            class_id = int(param_name.split('_')[1])  # Extract class ID from 'proxies_{class_id}'
            class_ids_to_register.append(class_id)
            
            if class_id not in self.member_bank:
                # Create the class entry manually without updating registered_classes yet
                device = param_value.device
                proxies = nn.Parameter(param_value.clone())
                
                self.member_bank[class_id] = {
                    'proxies': proxies,
                    'running_features': torch.zeros(self.num_proxies, self.feature_dim, device=device),
                    'feature_counts': torch.zeros(self.num_proxies, device=device)
                }
                
                # Register the parameter
                self.register_parameter(f'proxies_{class_id}', self.member_bank[class_id]['proxies'])
            
            # Load the proxy parameter
            self.member_bank[class_id]['proxies'].data.copy_(param_value)
        
        # Update registered_classes buffer with all loaded class IDs
        if class_ids_to_register:
            device = next(self.parameters()).device if hasattr(self, '_parameters') and self._parameters else 'cpu'
            self.registered_classes = torch.tensor(sorted(class_ids_to_register), device=device, dtype=torch.long)
            # print(f"JPL Module: Loaded proxies for {len(class_ids_to_register)} classes: {sorted(class_ids_to_register)}")
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to ensure all proxy parameters are included
        """
        # Get the standard state dict
        state_dict = super().state_dict(destination, prefix, keep_vars)
        
        # Ensure all proxy parameters are included
        for class_id in self.member_bank:
            param_name = f'{prefix}proxies_{class_id}'
            if param_name not in state_dict:
                if keep_vars:
                    state_dict[param_name] = self.member_bank[class_id]['proxies']
                else:
                    state_dict[param_name] = self.member_bank[class_id]['proxies'].data
        
        return state_dict
    
    def update_proxies(self, features, labels):
        """
        Update proxy representations based on new features
        
        Args:
            features (tensor): Feature tensor [batch_size, feature_dim]
            labels (tensor): Original label IDs [batch_size]
        """
        with torch.no_grad():
            # Normalize feature vectors
            features = F.normalize(features, p=2, dim=-1)
            
            unique_labels = labels.unique()
            
            for class_id in unique_labels:
                class_id_item = class_id.item()
                class_entry = self.member_bank[class_id_item]
                
                # Get features for this class
                class_mask = (labels == class_id)
                if not class_mask.any():
                    continue
                    
                class_features = features[class_mask]
                
                # Compute similarity between class features and class proxies
                # [batch_size_c, num_proxies]
                sim = torch.matmul(class_features, class_entry['proxies'].transpose(-2, -1))
                
                # Assign features to closest proxies (soft assignment)
                assignment = F.softmax(sim / self.temp, dim=-1)
                
                # For each proxy, compute the weighted average of assigned features
                for p in range(self.num_proxies):
                    weights = assignment[:, p].unsqueeze(-1)
                    weighted_sum = torch.sum(class_features * weights, dim=0)
                    weight_sum = weights.sum()
                    
                    if weight_sum > 0:
                        # Update running features
                        class_entry['running_features'][p] = (
                            class_entry['running_features'][p] * self.proxy_update_momentum + 
                            weighted_sum * (1 - self.proxy_update_momentum)
                        )
                        class_entry['feature_counts'][p] += weight_sum
                        
                        # Update proxy with the running average
                        proxy_update = F.normalize(class_entry['running_features'][p], p=2, dim=-1)
                        class_entry['proxies'].data[p] = (
                            class_entry['proxies'].data[p] * self.proxy_update_momentum + 
                            proxy_update * (1 - self.proxy_update_momentum)
                        )
                        class_entry['proxies'].data[p] = F.normalize(class_entry['proxies'].data[p], p=2, dim=-1)
    
    def random_masking(self, x):
        """
        Random masking for feature reconstruction task
        
        Args:
            x (tensor): Feature tensor with shape [batch_size, feature_dim]
            
        Returns:
            tuple: (masked_features, mask) 
                  - masked_features has same shape as x with masked values set to 0
                  - mask is boolean tensor with True at positions that are kept
        """
        B, D = x.shape
        len_keep = int(D * (1 - self.mask_ratio))
        
        noise = torch.rand(B, D, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore original order
        
        # Generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([B, D], device=x.device)
        mask[:, :len_keep] = 1
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Mask features
        masked_features = x * mask
        
        return masked_features, mask

    
    def local_reconstruction_loss_new(self, features, labels):
        """
        Compute local reconstruction loss with mask-aware proxy attention.
        
        Logic: 
        1. Original features -> masked -> encoder -> encoded features
        2. Compute attention between encoded features and corresponding proxies
        3. Attention-weighted proxy fusion -> decoder -> reconstruction
        4. MSE loss on masked positions only
        
        Args:
            features (tensor): Input features [B, D]
            labels (tensor): Ground truth class labels [B]
        
        Returns:
            Tensor: Reconstruction loss
        """
        batch_size, feat_dim = features.shape
        device = features.device
        
        masked_features, mask = self.random_masking(features)    # [B, D], [B, D]
        encoded_features = self.encoder(masked_features)         # [B, D]
        
        recon_loss = torch.tensor(0.0, device=device)
        
        for i in range(batch_size):
            label = labels[i].item()
            proxies = self.member_bank[label]['proxies']         # [K, D]
            original_feat = features[i]                          # [D]
            encoded_feat = encoded_features[i]                   # [D]
            mask_inv = ~mask[i].bool()                           # [D] (True for masked positions)
            
            if not mask_inv.any():
                continue  # Skip if no masked dimensions
            
            # Compute attention between encoded feature and proxies
            # Using cosine similarity or dot product
            similarities = torch.matmul(proxies, encoded_feat)   # [K]
            attention = F.softmax(similarities / self.temp, dim=0)  # [K]
            
            # Weighted proxy fusion
            combined_proxy = torch.matmul(attention, proxies)    # [D]
            
            # Decode to get reconstruction
            reconstructed = self.decoder(combined_proxy)         # [D]
            
            # Final MSE loss only on masked positions
            sample_loss = F.mse_loss(reconstructed[mask_inv], original_feat[mask_inv])
            recon_loss += sample_loss
        
        return recon_loss / batch_size

    
    def apply_margin(self, similarity, is_positive=True):
        """
        Apply margin to similarity scores
        
        Args:
            similarity (tensor): Similarity scores
            is_positive (bool): Whether these are positive similarities
            
        Returns:
            tensor: Similarity scores with margin applied
        """
        if is_positive:
            if self.margin_type == 'cos':
                # Cosine margin: subtract margin from positive similarities
                return similarity - self.margin
            elif self.margin_type == 'arc':
                # ArcFace margin: apply margin in angular space
                theta = torch.acos(torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7))
                return torch.cos(theta + self.margin)
        else:
            if self.margin_type == 'cos':
                # Add margin to negative similarities
                return similarity + self.margin
            elif self.margin_type == 'arc':
                # ArcFace margin: apply margin in angular space for negatives
                theta = torch.acos(torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7))
                return torch.cos(torch.clamp(theta - self.margin, min=0.0))
        
        return similarity  # Default: no margin
    
    def forward(self, features, labels=None, update_stats=True):
        """
        Forward pass of the JPL Module.
        
        Args:
            features (tensor): Feature tensor with shape [batch_size, feature_dim]
            labels (tensor): Original label IDs for the batch. Shape [batch_size]
                            If None, inference mode is used.
            update_stats (bool): Whether to update proxy statistics
                            
        Returns:
            dict: Dictionary containing:
                - 'proxy_loss': Proxy loss (ℒProx)
                - 'recon_loss': Local reconstruction loss
                - 'total_loss': Combined loss
                - 'proxy_feat': Proxy features (for visualization/analysis)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Normalize feature vectors for cosine similarity
        normalized_features = F.normalize(features, p=2, dim=-1)
        
        if self.training:
            # Register new labels if any
            self.register_new_labels(labels, device)
            
            # Extract all proxies for the active classes
            all_proxies = []
            for class_id in self.registered_classes:
                all_proxies.append(self.member_bank[class_id.item()]['proxies'])
            
            # Stack proxies for visualization/analysis
            stacked_proxies = torch.cat(all_proxies, dim=0)  # [num_registered_classes * num_proxies, feature_dim]
            
            num_registered_classes = len(self.registered_classes)
            
            if num_registered_classes > 0:
                # Compute similarities for each class
                all_similarities = []
                
                # Compute similarity between features and all class proxies
                for i, class_id in enumerate(self.registered_classes):
                    # Get proxies for this class
                    proxies = self.member_bank[class_id.item()]['proxies']  # [num_proxies, feature_dim]
                    
                    # Compute cosine similarity: [batch_size, num_proxies]
                    class_similarity = torch.matmul(normalized_features, proxies.t())
                    
                    # Append to list
                    all_similarities.append(class_similarity)
                
                # Stack class similarities: [batch_size, num_registered_classes, num_proxies]
                stacked_similarities = torch.stack([s.unsqueeze(1) for s in all_similarities], dim=1)
                
                # Compute proxy loss with margin
                proxy_loss = torch.tensor(0.0, device=device)
                
                for i, sample_idx in enumerate(range(batch_size)):
                    label = labels[sample_idx].item()
                    
                    # Find the index of this label in registered_classes
                    label_idx = (self.registered_classes == label).nonzero().item()
                    
                    # Get positive similarities
                    pos_similarities = stacked_similarities[sample_idx, label_idx]
                    
                    # Apply margin to positive similarities
                    pos_similarities_with_margin = self.apply_margin(pos_similarities, is_positive=True)
                    
                    if num_registered_classes > 1:
                        # Create mask for negative classes
                        neg_mask = torch.ones(num_registered_classes, device=device).bool()
                        neg_mask[label_idx] = False
                        
                        # Get negative similarities
                        neg_similarities = stacked_similarities[sample_idx, neg_mask].reshape(-1)
                        
                        # Apply margin to negative similarities
                        neg_similarities_with_margin = self.apply_margin(neg_similarities, is_positive=False)
                        
                        # Compute loss for this sample
                        pos_term = -torch.mean(pos_similarities_with_margin)
                        neg_term = torch.mean(neg_similarities_with_margin) / (num_registered_classes - 1)
                        sample_loss = pos_term + neg_term
                    else:
                        # If only one class, just optimize positive similarities
                        sample_loss = -torch.mean(pos_similarities_with_margin)
                    
                    proxy_loss += sample_loss
                
                proxy_loss /= batch_size

                # FRL
                if self.use_local_recon:
                    # Compute local reconstruction loss
                    # recon_loss = self.local_reconstruction_loss(features, labels)
                    recon_loss = self.local_reconstruction_loss_new(features, labels)
                else:
                    recon_loss = torch.tensor(0.0, device=device)
                
                # Combine losses
                total_loss = proxy_loss + self.recon_weight * recon_loss
                
            else:
                proxy_loss = torch.tensor(0.0, device=device)
                recon_loss = torch.tensor(0.0, device=device)
                total_loss = torch.tensor(0.0, device=device)
                stacked_proxies = torch.zeros(0, self.feature_dim, device=device)
            
            # Update proxies if needed
            if update_stats and self.training:
                self.update_proxies(normalized_features, labels)
            
            return {
                'proxy_loss': proxy_loss,
                'recon_loss': recon_loss,
                'total_loss': total_loss,
                'proxy_feat': stacked_proxies
            }
        else:
            # Inference mode
            proxies = []
            # Compute similarity between features and all class proxies
            for i, class_id in enumerate(self.registered_classes):
                # Get proxies for this class
                proxy = self.member_bank[class_id.item()]['proxies']  # [num_proxies, feature_dim]
                proxies.append(proxy)
            
            if len(proxies) > 0:
                proxies = torch.cat(proxies, dim=0)  # [num_registered_classes * num_proxies, feature_dim]
            else:
                proxies = torch.zeros(0, self.feature_dim, device=device)
            return proxies
    
    def get_registered_classes(self):
        """
        Get the list of registered class IDs
        
        Returns:
            list: List of registered class IDs
        """
        return self.registered_classes.tolist()

    def get_proxies(self):
        """
        Get all registered proxies
        
        Returns:
            torch.Tensor: Concatenated proxies from all registered classes
        """
        if len(self.registered_classes) == 0:
            # If no classes registered, try to infer from member_bank
            if self.member_bank:
                class_ids = sorted(self.member_bank.keys())
                device = next(self.parameters()).device if hasattr(self, '_parameters') and self._parameters else 'cpu'
                self.registered_classes = torch.tensor(class_ids, device=device, dtype=torch.long)
                print(f"JPL Module: Auto-registered classes from member_bank: {class_ids}")
            else:
                # Return empty tensor if no proxies available
                device = next(self.parameters()).device if hasattr(self, '_parameters') and self._parameters else 'cpu'
                return torch.zeros(0, self.feature_dim, device=device)
        
        proxies = []
        for class_id in self.registered_classes:
            class_id_item = class_id.item()
            if class_id_item in self.member_bank:
                proxy = self.member_bank[class_id_item]['proxies']  # [num_proxies, feature_dim]
                proxies.append(proxy.reshape(1,-1))
            else:
                print(f"Warning: Class {class_id_item} in registered_classes but not in member_bank")
        
        if len(proxies) > 0:
            proxies = torch.cat(proxies, dim=0)  # [num_registered_classes * num_proxies, feature_dim]
        else:
            device = next(self.parameters()).device if hasattr(self, '_parameters') and self._parameters else 'cpu'
            proxies = torch.zeros(0, self.feature_dim, device=device)
        
        return proxies
        
    def get_num_registered_classes(self):
        """
        Get the number of registered classes
        
        Returns:
            int: Number of registered classes
        """
        return len(self.registered_classes)