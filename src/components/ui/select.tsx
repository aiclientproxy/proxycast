import React, { createContext, useContext, useState } from "react";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";

interface SelectContextType {
  value: string;
  onValueChange: (value: string) => void;
  open: boolean;
  setOpen: (open: boolean) => void;
  disabled: boolean;
}

const SelectContext = createContext<SelectContextType | undefined>(undefined);

interface SelectProps {
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  disabled?: boolean;
  children: React.ReactNode;
  closeOnMouseLeave?: boolean;
}

const Select: React.FC<SelectProps> = ({
  value,
  defaultValue,
  onValueChange,
  disabled = false,
  children,
  closeOnMouseLeave = false,
}) => {
  const [internalValue, setInternalValue] = useState(defaultValue || "");
  const [open, setOpen] = useState(false);

  const currentValue = value !== undefined ? value : internalValue;
  const handleValueChange = onValueChange || setInternalValue;

  return (
    <SelectContext.Provider
      value={{
        value: currentValue,
        onValueChange: handleValueChange,
        open,
        setOpen,
        disabled,
      }}
    >
      <div
        className="relative"
        onMouseLeave={closeOnMouseLeave ? () => setOpen(false) : undefined}
      >
        {children}
      </div>
    </SelectContext.Provider>
  );
};

interface SelectTriggerProps {
  className?: string;
  children: React.ReactNode;
}

const SelectTrigger: React.FC<SelectTriggerProps> = ({
  className,
  children,
}) => {
  const context = useContext(SelectContext);
  if (!context) throw new Error("SelectTrigger must be used within Select");

  const { open, setOpen, disabled } = context;

  return (
    <button
      type="button"
      disabled={disabled}
      className={cn(
        "flex h-10 w-full items-center justify-between rounded-md border border-gray-300 bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
        className,
      )}
      onClick={() => !disabled && setOpen(!open)}
    >
      {children}
      <ChevronDown className="h-4 w-4 opacity-50" />
    </button>
  );
};

interface SelectValueProps {
  placeholder?: string;
  className?: string;
}

const SelectValue: React.FC<SelectValueProps> = ({ placeholder }) => {
  const context = useContext(SelectContext);
  if (!context) throw new Error("SelectValue must be used within Select");

  const { value } = context;
  return <span>{value || placeholder}</span>;
};

interface SelectContentProps {
  className?: string;
  children: React.ReactNode;
}

const SelectContent: React.FC<SelectContentProps> = ({
  className,
  children,
}) => {
  const context = useContext(SelectContext);
  if (!context) throw new Error("SelectContent must be used within Select");

  const { open } = context;

  if (!open) return null;

  return (
    <div
      className={cn(
        "absolute top-full z-50 w-full rounded-md border bg-white shadow-md",
        className,
      )}
    >
      {children}
    </div>
  );
};

interface SelectItemProps {
  value: string;
  className?: string;
  children: React.ReactNode;
}

const SelectItem: React.FC<SelectItemProps> = ({
  value,
  className,
  children,
}) => {
  const context = useContext(SelectContext);
  if (!context) throw new Error("SelectItem must be used within Select");

  const { onValueChange, setOpen, value: selectedValue } = context;

  const handleSelect = () => {
    onValueChange(value);
    setOpen(false);
  };

  const isSelected = selectedValue === value;

  return (
    <div
      className={cn(
        "relative flex cursor-default select-none items-center justify-between rounded-sm px-2 py-2 text-sm outline-none transition-colors hover:bg-accent hover:text-accent-foreground data-[state=checked]:bg-accent/50",
        isSelected && "bg-accent/50 text-accent-foreground",
        className,
      )}
      onClick={handleSelect}
    >
      {children}
      {isSelected && (
        <span className="flex h-3.5 w-3.5 items-center justify-center ml-2">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-4 w-4 opacity-100" // Always visible if selected
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </span>
      )}
    </div>
  );
};

export { Select, SelectContent, SelectItem, SelectTrigger, SelectValue };
