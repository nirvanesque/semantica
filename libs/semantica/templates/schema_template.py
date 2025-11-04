"""
Schema Template for Semantica framework.

Provides template-based knowledge graph construction
to prevent AI from inventing new entities or relationships.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


@dataclass
class EntityTemplate:
    """Entity template definition."""
    
    name: str
    entity_type: str
    required_properties: List[str] = field(default_factory=list)
    optional_properties: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipTemplate:
    """Relationship template definition."""
    
    name: str
    subject_type: str
    object_type: str
    predicate: str
    cardinality: str = "many-to-many"  # one-to-one, one-to-many, many-to-many
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaTemplate:
    """Schema template for knowledge graph."""
    
    name: str
    namespace: str
    entity_templates: Dict[str, EntityTemplate] = field(default_factory=dict)
    relationship_templates: Dict[str, RelationshipTemplate] = field(default_factory=dict)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchemaTemplateManager:
    """
    Schema template management and validation.
    
    • Defines and validates entity templates
    • Manages relationship constraints
    • Enforces template-based extraction
    • Validates extracted entities against templates
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize schema template manager."""
        self.logger = get_logger("schema_template_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.templates: Dict[str, SchemaTemplate] = {}
        self.active_template: Optional[str] = None
    
    def create_template(
        self,
        name: str,
        namespace: str,
        entity_templates: Optional[List[EntityTemplate]] = None,
        relationship_templates: Optional[List[RelationshipTemplate]] = None,
        **metadata
    ) -> SchemaTemplate:
        """
        Create a new schema template.
        
        Args:
            name: Template name
            namespace: Namespace URI
            entity_templates: List of entity templates
            relationship_templates: List of relationship templates
            **metadata: Additional metadata
            
        Returns:
            Created SchemaTemplate
        """
        template = SchemaTemplate(
            name=name,
            namespace=namespace,
            entity_templates={e.name: e for e in (entity_templates or [])},
            relationship_templates={r.name: r for r in (relationship_templates or [])},
            metadata=metadata
        )
        
        self.templates[name] = template
        self.logger.info(f"Created schema template: {name}")
        
        return template
    
    def add_entity_template(
        self,
        template_name: str,
        entity_template: EntityTemplate
    ) -> bool:
        """Add entity template to schema."""
        template = self.templates.get(template_name)
        if not template:
            raise ValidationError(f"Template '{template_name}' not found")
        
        template.entity_templates[entity_template.name] = entity_template
        return True
    
    def add_relationship_template(
        self,
        template_name: str,
        relationship_template: RelationshipTemplate
    ) -> bool:
        """Add relationship template to schema."""
        template = self.templates.get(template_name)
        if not template:
            raise ValidationError(f"Template '{template_name}' not found")
        
        template.relationship_templates[relationship_template.name] = relationship_template
        return True
    
    def validate_entity(
        self,
        entity: Dict[str, Any],
        template_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate entity against template.
        
        Args:
            entity: Entity dictionary
            template_name: Template name (uses active template if None)
            
        Returns:
            Validation result
        """
        template_name = template_name or self.active_template
        if not template_name:
            return {"valid": True, "warnings": ["No template active"]}
        
        template = self.templates.get(template_name)
        if not template:
            return {"valid": False, "errors": [f"Template '{template_name}' not found"]}
        
        entity_type = entity.get("type")
        if not entity_type:
            return {"valid": False, "errors": ["Entity type missing"]}
        
        # Find matching entity template
        entity_template = None
        for et in template.entity_templates.values():
            if et.entity_type == entity_type:
                entity_template = et
                break
        
        if not entity_template:
            return {"valid": False, "errors": [f"No template for entity type '{entity_type}'"]}
        
        # Validate required properties
        errors = []
        for prop in entity_template.required_properties:
            if prop not in entity or entity[prop] is None:
                errors.append(f"Missing required property: {prop}")
        
        # Check constraints
        for prop, constraint in entity_template.constraints.items():
            if prop in entity:
                value = entity[prop]
                if not self._check_constraint(value, constraint):
                    errors.append(f"Constraint violation for property '{prop}': {constraint}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "entity_template": entity_template.name
        }
    
    def validate_relationship(
        self,
        relationship: Dict[str, Any],
        template_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate relationship against template.
        
        Args:
            relationship: Relationship dictionary
            template_name: Template name (uses active template if None)
            
        Returns:
            Validation result
        """
        template_name = template_name or self.active_template
        if not template_name:
            return {"valid": True, "warnings": ["No template active"]}
        
        template = self.templates.get(template_name)
        if not template:
            return {"valid": False, "errors": [f"Template '{template_name}' not found"]}
        
        predicate = relationship.get("predicate")
        subject_type = relationship.get("subject_type")
        object_type = relationship.get("object_type")
        
        if not predicate:
            return {"valid": False, "errors": ["Relationship predicate missing"]}
        
        # Find matching relationship template
        rel_template = None
        for rt in template.relationship_templates.values():
            if (rt.predicate == predicate and
                rt.subject_type == subject_type and
                rt.object_type == object_type):
                rel_template = rt
                break
        
        if not rel_template:
            return {
                "valid": False,
                "errors": [f"No template for relationship '{predicate}' between {subject_type} and {object_type}"]
            }
        
        # Check constraints
        errors = []
        for prop, constraint in rel_template.constraints.items():
            if prop in relationship:
                value = relationship[prop]
                if not self._check_constraint(value, constraint):
                    errors.append(f"Constraint violation for property '{prop}': {constraint}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "relationship_template": rel_template.name
        }
    
    def get_allowed_entities(self, template_name: Optional[str] = None) -> List[str]:
        """Get list of allowed entity types."""
        template_name = template_name or self.active_template
        if not template_name:
            return []
        
        template = self.templates.get(template_name)
        if not template:
            return []
        
        return [et.entity_type for et in template.entity_templates.values()]
    
    def get_allowed_relationships(
        self,
        template_name: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Get list of allowed relationships."""
        template_name = template_name or self.active_template
        if not template_name:
            return []
        
        template = self.templates.get(template_name)
        if not template:
            return []
        
        return [
            {
                "predicate": rt.predicate,
                "subject_type": rt.subject_type,
                "object_type": rt.object_type
            }
            for rt in template.relationship_templates.values()
        ]
    
    def set_active_template(self, template_name: str) -> bool:
        """Set active template for validation."""
        if template_name not in self.templates:
            raise ValidationError(f"Template '{template_name}' not found")
        
        self.active_template = template_name
        return True
    
    def _check_constraint(self, value: Any, constraint: Any) -> bool:
        """Check if value satisfies constraint."""
        if isinstance(constraint, dict):
            if "type" in constraint:
                expected_type = constraint["type"]
                if expected_type == "string" and not isinstance(value, str):
                    return False
                if expected_type == "number" and not isinstance(value, (int, float)):
                    return False
            
            if "min" in constraint and isinstance(value, (int, float)):
                if value < constraint["min"]:
                    return False
            
            if "max" in constraint and isinstance(value, (int, float)):
                if value > constraint["max"]:
                    return False
        
        return True
